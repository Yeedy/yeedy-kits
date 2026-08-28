import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
import rolebench


class RoleBenchTests(unittest.TestCase):
    def make_run_args(self, root, **overrides):
        config = root / "config.yml"
        config.write_text("modelRoles:\n  smol: p/base\n  task: p/base\n", encoding="utf-8")
        values = {
            "roles": "task",
            "config_file": str(config),
            "models": "p/a",
            "model": None,
            "providers": "all",
            "catalog_file": None,
            "omp_bin": "omp",
            "mode": "role-isolated",
            "assignment_file": None,
            "suite": "none",
            "tasks_file": None,
            "runs": 1,
            "execute": True,
            "executor": "direct",
            "controller_model": None,
            "confirm": True,
            "pricing_file": None,
            "max_cost_usd": None,
            "timeout": 30,
            "seed": 0,
            "output": str(root / "run"),
        }
        values.update(overrides)
        return SimpleNamespace(**values)

    def test_selector_keeps_thinking_separate(self):
        self.assertEqual(
            rolebench.parse_selector("opencode-go/glm-5.3-flash:high"),
            {
                "provider": "opencode-go",
                "model": "glm-5.3-flash",
                "selector": "opencode-go/glm-5.3-flash",
                "thinking": "high",
            },
        )

    def test_cost_extraction_does_not_confuse_token_total(self):
        self.assertEqual(rolebench.extract_cost({"input": 2, "output": 3, "cost": {"total": 0.012}}), 0.012)
        self.assertIsNone(rolebench.extract_cost({"input": 2, "output": 3, "total": 5}))

    def test_explicit_thinking_variants_remain_distinct(self):
        args = SimpleNamespace(models="p/a:high,p/a:low", model=None, providers="all", mode="role-isolated", assignment_file=None, omp_bin="missing", catalog_file=None)
        candidates, _ = rolebench.resolve_candidates(args)
        self.assertEqual(len(candidates), 2)
        self.assertEqual({candidate["thinking"] for candidate in candidates}, {"high", "low"})

    def test_estimate_uses_explicit_per_million_rates(self):
        pricing_path = Path(__file__).parent / "fixtures" / "pricing.json"
        pricing = rolebench.load_pricing(str(pricing_path))
        candidate = rolebench.parse_selector("fake/model")
        assignment = {"id": "a", "mode": "role-isolated", "role": "task", "candidate": "fake/model", "mapping": {"task": "fake/model"}}
        task = {"id": "task-001", "role": "task", "prompt": "do it", "max_output_tokens": 100}
        estimate = rolebench.estimate_benchmark(["task"], [assignment], [task], 2, pricing)
        self.assertEqual(estimate["calls"], 2)
        self.assertEqual(estimate["cost_status"], "estimated")
        self.assertGreater(estimate["cost_usd"], 0)

    def test_catalog_normalizes_provider_group(self):
        fixture = Path(__file__).parent / "fixtures" / "catalog.json"
        catalog = rolebench.normalize_catalog(rolebench.load_json_document(fixture))
        self.assertEqual(len(catalog), 3)
        self.assertEqual(catalog[0]["selector"], "kimi-code/k3-256k")
        grouped = rolebench.normalize_catalog({"models": {"opencode-go": [{"id": "glm-5.3-flash"}]}})
        self.assertEqual(grouped[0]["selector"], "opencode-go/glm-5.3-flash")

    def test_role_isolated_and_all_same_matrix(self):
        candidates = [rolebench.parse_selector("p/a"), rolebench.parse_selector("p/b")]
        isolated = rolebench.build_assignments("role-isolated", ["smol", "task"], {"smol": "p/base", "task": "p/base"}, candidates, None)
        all_same = rolebench.build_assignments("all-same", ["smol", "task"], {}, candidates, None)
        self.assertEqual(len(isolated), 4)
        self.assertEqual(isolated[0]["mapping"]["task"], "p/base")
        self.assertEqual(isolated[1]["mapping"]["smol"], "p/base")
        self.assertEqual(len(all_same), 2)
        self.assertEqual(all_same[0]["mapping"], {"smol": "p/a", "task": "p/a"})

    def test_explicit_assignment_does_not_need_catalog(self):
        path = Path(__file__).parent / "fixtures" / "assignment.yml"
        args = SimpleNamespace(models=None, model=None, providers="all", catalog_file=None, omp_bin="missing", mode="explicit", assignment_file=str(path))
        candidates, source = rolebench.resolve_candidates(args)
        self.assertEqual(source["source"], "assignment")
        self.assertEqual({candidate["thinking"] for candidate in candidates}, {"high"})

    @patch("rolebench.subprocess.run")
    def test_direct_run_records_output(self, run):
        run.return_value = SimpleNamespace(returncode=0, stdout=json.dumps({"text": "PASS"}), stderr="")
        assignment = {"id": "a", "mode": "role-isolated", "role": "task", "candidate": "p/a", "mapping": {"task": "p/a"}, "thinking": {"task": "high"}}
        task = {"id": "task-001", "role": "task", "prompt": "do it"}
        row = rolebench.run_one("omp", assignment, task, 1, Path("."), 5, Path("."), "direct", None)
        self.assertEqual(row["status"], "pass")
        self.assertEqual(row["output"], "PASS")
        self.assertIn("--thinking", run.call_args.args[0])
        self.assertEqual(run.call_args.args[0][run.call_args.args[0].index("--thinking") + 1], "high")
        self.assertEqual(run.call_args.kwargs["errors"], "replace")

    @patch("rolebench.subprocess.run")
    def test_delegated_run_uses_project_probe_agent(self, run):
        run.return_value = SimpleNamespace(returncode=0, stdout=json.dumps({"text": "<task-result status=\"completed\">PASS</task-result>"}), stderr="")
        assignment = {"id": "a", "mode": "role-isolated", "role": "task", "candidate": "p/a", "mapping": {"task": "p/a"}, "thinking": {"task": "high"}}
        task = {"id": "task-001", "role": "task", "prompt": "do it"}
        row = rolebench.run_one("omp", assignment, task, 1, Path("."), 5, Path("."), "delegated", "p/controller")
        self.assertEqual(row["status"], "pass")
        command = run.call_args.args[0]
        self.assertIn("--tools", command)
        self.assertIn("task", command)

    @patch("rolebench.subprocess.run")
    def test_delegated_run_defaults_controller_without_unbound_error(self, run):
        run.return_value = SimpleNamespace(returncode=0, stdout=json.dumps({"text": "PASS"}), stderr="")
        assignment = {"id": "a", "mode": "role-isolated", "role": "task", "candidate": "p/a", "mapping": {"task": "p/a"}, "thinking": {"task": "high"}}
        task = {"id": "task-001", "role": "task", "prompt": "do it"}
        row = rolebench.run_one("omp", assignment, task, 1, Path("."), 5, Path("."), "delegated", None)
        self.assertEqual(row["status"], "pass")
        self.assertEqual(row["controller_model"], "p/a")

    @patch("rolebench.subprocess.run")
    def test_timeout_output_is_text_and_json_serializable(self, run):
        run.side_effect = subprocess.TimeoutExpired(["omp"], 1, output=b"partial\xff", stderr=b"err")
        assignment = {"id": "a", "mode": "role-isolated", "role": "task", "candidate": "p/a", "mapping": {"task": "p/a"}}
        task = {"id": "task-001", "role": "task", "prompt": "do it"}
        row = rolebench.run_one("omp", assignment, task, 1, Path("."), 1, Path("."), "direct", None)
        self.assertEqual(row["error_type"], "timeout")
        self.assertEqual(row["stdout"], "partial�")
        self.assertEqual(row["stderr"], "err")
        self.assertTrue(row["provider_call_started"])
        json.dumps(row)

    @patch("rolebench.subprocess.run")
    def test_non_utf8_output_is_decoded_as_replacement_text(self, run):
        run.return_value = SimpleNamespace(returncode=0, stdout=b'{"text":"ok\xff"}', stderr=b"")
        assignment = {"id": "a", "mode": "role-isolated", "role": "task", "candidate": "p/a", "mapping": {"task": "p/a"}}
        task = {"id": "task-001", "role": "task", "prompt": "do it"}
        row = rolebench.run_one("omp", assignment, task, 1, Path("."), 1, Path("."), "direct", None)
        self.assertEqual(row["status"], "pass")
        self.assertEqual(row["output"], "ok�")

    @patch("rolebench.subprocess.run")
    def test_run_one_preserves_provider_error_rows(self, run):
        assignment = {"id": "a", "mode": "role-isolated", "role": "task", "candidate": "p/a", "mapping": {"task": "p/a"}}
        task = {"id": "task-001", "role": "task", "prompt": "do it"}
        cases = [
            (SimpleNamespace(returncode=7, stdout="partial", stderr="provider failed"), "omp_exit"),
            (SimpleNamespace(returncode=0, stdout="not json", stderr=""), "malformed_omp_json"),
            (SimpleNamespace(returncode=0, stdout=json.dumps({"text": ""}), stderr=""), "empty_output"),
        ]
        for completed, error_type in cases:
            with self.subTest(error_type=error_type):
                run.return_value = completed
                row = rolebench.run_one("omp", assignment, task, 1, Path("."), 1, Path("."), "direct", None)
                self.assertEqual(row["status"], "error")
                self.assertEqual(row["error_type"], error_type)
                self.assertTrue(row["provider_call_started"])
                json.dumps(row)

    def test_missing_config_is_rejected_for_role_isolated(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = self.make_run_args(root, config_file=str(root / "missing.yml"))
            with self.assertRaises(rolebench.RoleBenchError):
                rolebench.run_command(args)

    def test_missing_config_is_allowed_for_explicit_mode(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            assignment = root / "assignment.yml"
            assignment.write_text("assignment:\n  task: p/a\n", encoding="utf-8")
            args = self.make_run_args(root, mode="explicit", roles="task", config_file=str(root / "missing.yml"), models=None, assignment_file=str(assignment), execute=False, confirm=False)
            with patch.object(rolebench, "load_tasks", return_value=[{"id": "task-1", "role": "task", "prompt": "do"}]):
                self.assertEqual(rolebench.run_command(args), 0)

    def test_incremental_results_survive_interruption(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = self.make_run_args(root, runs=2)
            first = {"status": "pass", "provider_call_started": True, "output": "first"}
            with patch.object(rolebench, "load_tasks", return_value=[{"id": "task-1", "role": "task", "prompt": "do"}]):
                with patch.object(rolebench, "run_one", side_effect=[first, KeyboardInterrupt]):
                    with self.assertRaises(KeyboardInterrupt):
                        rolebench.run_command(args)
            lines = (root / "run" / "results.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 1)
            self.assertEqual(json.loads(lines[0])["output"], "first")

    def test_timeout_stops_capped_run_as_unverifiable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pricing = root / "pricing.json"
            pricing.write_text(json.dumps({"p/a": {"input_per_million": 1, "output_per_million": 1}}), encoding="utf-8")
            args = self.make_run_args(root, runs=2, pricing_file=str(pricing), max_cost_usd=0.5)
            timeout_row = {"status": "error", "error_type": "timeout", "provider_call_started": True}
            pass_row = {"status": "pass", "provider_call_started": True, "exit_code": 0, "usage": {"cost": {"total": 0.001}}}
            with patch.object(rolebench, "load_tasks", return_value=[{"id": "task-1", "role": "task", "prompt": "do"}]):
                with patch.object(rolebench, "run_one", side_effect=[timeout_row, pass_row]) as run:
                    self.assertEqual(rolebench.run_command(args), 0)
            summary = json.loads((root / "run" / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(run.call_count, 1)
            self.assertEqual(summary["budget_status"], "unverifiable")
            self.assertEqual(summary["stopped_reason"], "provider_did_not_return_cost")

    def test_actual_cost_stops_capped_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pricing = root / "pricing.json"
            pricing.write_text(json.dumps({"p/a": {"input_per_million": 1, "output_per_million": 1}}), encoding="utf-8")
            args = self.make_run_args(root, runs=3, pricing_file=str(pricing), max_cost_usd=0.0019)
            first = {"status": "pass", "provider_call_started": True, "exit_code": 0, "usage": {"cost": {"total": 0.001}}}
            second = {"status": "pass", "provider_call_started": True, "exit_code": 0, "usage": {"cost": {"total": 0.001}}}
            with patch.object(rolebench, "load_tasks", return_value=[{"id": "task-1", "role": "task", "prompt": "do"}]):
                with patch.object(rolebench, "run_one", side_effect=[first, second]) as run:
                    self.assertEqual(rolebench.run_command(args), 0)
            summary = json.loads((root / "run" / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(run.call_count, 2)
            self.assertEqual(summary["budget_status"], "reached")
            self.assertEqual(summary["stopped_reason"], "actual_cost_reached_cap")

    def test_unknown_cost_blocks_capped_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = self.make_run_args(root, max_cost_usd=0.5)
            with patch.object(rolebench, "load_tasks", return_value=[{"id": "task-1", "role": "task", "prompt": "do"}]):
                with patch.object(rolebench, "run_one") as run:
                    self.assertEqual(rolebench.run_command(args), 2)
            run.assert_not_called()
            summary = json.loads((root / "run" / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["reason"], "cost_unknown")

    def test_cli_rejects_invalid_run_limits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "config.yml"
            config.write_text("modelRoles:\n  task: p/base\n", encoding="utf-8")
            common = ["run", "--models", "p/a", "--roles", "task", "--config-file", str(config), "--output", str(root / "run")]
            for option, value in (("--runs", "0"), ("--timeout", "0"), ("--max-cost-usd", "-1")):
                self.assertEqual(rolebench.main(common + [option, value]), 2)

    def test_stale_results_are_rotated_on_rerun(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "config.yml"
            config.write_text("modelRoles:\n  task: p/base\n", encoding="utf-8")
            output = root / "run"
            output.mkdir()
            old = output / "results.jsonl"
            old.write_text('{"role":"task","model":"p/old"}\n', encoding="utf-8")
            self.assertEqual(rolebench.main(["run", "--models", "p/new", "--roles", "task", "--config-file", str(config), "--output", str(output)]), 0)
            self.assertFalse(old.exists())
            self.assertEqual(len(list(output.glob("results.jsonl.previous-*"))), 1)

    def test_discover_uses_catalog_fixture(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "catalog.json"
            fixture = Path(__file__).parent / "fixtures" / "catalog.json"
            self.assertEqual(rolebench.main(["discover", "--catalog-file", str(fixture), "--output", str(output)]), 0)
            self.assertEqual(json.loads(output.read_text(encoding="utf-8"))["count"], 3)

    def test_report_json_honors_output_and_thinking_variants(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "manifest.json").write_text(json.dumps({"generated_at": "now", "mode": "role-isolated", "roles": ["task"], "assignments": []}), encoding="utf-8")
            rows = [
                {"role": "task", "model": "p/a", "thinking": "high", "status": "pass", "latency_ms": 1, "output": "high"},
                {"role": "task", "model": "p/a", "thinking": "low", "status": "pass", "latency_ms": 2, "output": "low"},
            ]
            (root / "results.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
            output = root / "report.json"
            self.assertEqual(rolebench.main(["report", "--run", str(root), "--format", "json", "--output", str(output)]), 0)
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertIn("task/p/a:high", payload["groups"])
            self.assertIn("task/p/a:low", payload["groups"])

    def test_report_markdown_closes_thinking_cell(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "manifest.json").write_text(json.dumps({"generated_at": "now", "mode": "role-isolated", "roles": ["task"], "assignments": []}), encoding="utf-8")
            row = {"role": "task", "model": "p/a", "thinking": "high", "status": "pass", "latency_ms": 1, "output": "ok"}
            (root / "results.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
            output = root / "report.md"
            self.assertEqual(rolebench.main(["report", "--run", str(root), "--format", "markdown", "--output", str(output)]), 0)
            report = output.read_text(encoding="utf-8")
            self.assertIn("| `task` | `p/a` | `high` |", report)

    def test_fake_executor_end_to_end(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "config.yml"
            config.write_text("modelRoles:\n  task: fake/model\n", encoding="utf-8")
            tasks = root / "tasks.json"
            tasks.write_text(json.dumps([{"id": "task-1", "role": "task", "prompt": "do"}]), encoding="utf-8")
            output = root / "run"
            fake_omp = (Path(__file__).parent / "fake_omp").resolve()
            self.assertEqual(rolebench.main([
                "run", "--omp-bin", str(fake_omp), "--models", "fake/model", "--roles", "task",
                "--config-file", str(config), "--tasks-file", str(tasks), "--execute", "--confirm", "--output", str(output),
            ]), 0)
            result = json.loads((output / "results.jsonl").read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(result["status"], "pass")

    @patch("rolebench.subprocess.run")
    def test_execute_requires_confirmation_before_provider_call(self, run):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "config.yml"
            config.write_text("modelRoles:\n  task: fake/model\n", encoding="utf-8")
            output = root / "run"
            code = rolebench.main([
                "run",
                "--models", "fake/model",
                "--roles", "task",
                "--config-file", str(config),
                "--execute",
                "--output", str(output),
            ])
            self.assertEqual(code, 3)
            run.assert_not_called()
            summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["status"], "awaiting_consent")

    @patch("rolebench.subprocess.run")
    def test_cost_cap_blocks_before_provider_call(self, run):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "config.yml"
            config.write_text("modelRoles:\n  task: fake/model\n", encoding="utf-8")
            pricing = root / "pricing.json"
            pricing.write_text(json.dumps({"fake/model": {"input_per_million": 1, "output_per_million": 1}}), encoding="utf-8")
            output = root / "run"
            code = rolebench.main([
                "run",
                "--models", "fake/model",
                "--roles", "task",
                "--config-file", str(config),
                "--pricing-file", str(pricing),
                "--max-cost-usd", "0",
                "--execute",
                "--confirm",
                "--output", str(output),
            ])
            self.assertEqual(code, 2)
            run.assert_not_called()
            summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["reason"], "estimated_cost_exceeds_cap")


if __name__ == "__main__":
    unittest.main()
