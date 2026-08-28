import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parents[1] / "scripts"))
import rolebench


class RoleBenchTests(unittest.TestCase):
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

    def test_estimate_uses_explicit_per_million_rates(self):
        candidate = rolebench.parse_selector("fake/model")
        assignment = {"id": "a", "mode": "role-isolated", "role": "task", "candidate": "fake/model", "mapping": {"task": "fake/model"}}
        task = {"id": "task-001", "role": "task", "prompt": "do it", "max_output_tokens": 100}
        estimate = rolebench.estimate_benchmark(["task"], [assignment], [task], 2, {candidate["selector"]: {"input_per_million": 1, "output_per_million": 2}})
        self.assertEqual(estimate["calls"], 2)
        self.assertEqual(estimate["cost_status"], "estimated")
        self.assertGreater(estimate["cost_usd"], 0)

    def test_catalog_normalizes_provider_group(self):
        catalog = rolebench.normalize_catalog({"models": {"opencode-go": [{"id": "glm-5.3-flash"}]}})
        self.assertEqual(catalog[0]["selector"], "opencode-go/glm-5.3-flash")

    def test_role_isolated_and_all_same_matrix(self):
        candidates = [rolebench.parse_selector("p/a"), rolebench.parse_selector("p/b")]
        isolated = rolebench.build_assignments("role-isolated", ["smol", "task"], {"smol": "p/base", "task": "p/base"}, candidates, None)
        all_same = rolebench.build_assignments("all-same", ["smol", "task"], {}, candidates, None)
        self.assertEqual(len(isolated), 4)
        self.assertEqual(len(all_same), 2)
        self.assertEqual(all_same[0]["mapping"], {"smol": "p/a", "task": "p/a"})

    def test_explicit_assignment_does_not_need_catalog(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "assignment.yml"
            path.write_text("assignment:\n  task: p/a:high\n", encoding="utf-8")
            args = SimpleNamespace(models=None, model=None, providers="all", catalog_file=None, omp_bin="missing", mode="explicit", assignment_file=str(path))
            candidates, source = rolebench.resolve_candidates(args)
            self.assertEqual(source["source"], "assignment")
            self.assertEqual(candidates[0]["thinking"], "high")

    @patch("rolebench.subprocess.run")
    def test_direct_run_records_output(self, run):
        run.return_value = SimpleNamespace(returncode=0, stdout=json.dumps({"text": "PASS"}), stderr="")
        assignment = {"id": "a", "mode": "role-isolated", "role": "task", "candidate": "p/a", "mapping": {"task": "p/a"}, "thinking": {"task": "high"}}
        task = {"id": "task-001", "role": "task", "prompt": "do it"}
        row = rolebench.run_one("omp", assignment, task, 1, Path("."), 5, Path("."), "direct", None)
        self.assertEqual(row["status"], "pass")
        self.assertEqual(row["output"], "PASS")
        self.assertIn("--thinking", run.call_args.args[0])

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
