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
    def git(self, directory, *arguments):
        return subprocess.run(
            ["git", "-C", str(directory), *arguments],
            check=True,
            capture_output=True,
            text=True,
        )

    def init_git_repo(self, root):
        self.git(root, "init", "-q")
        self.git(root, "config", "user.email", "rolebench-tests@example.com")
        self.git(root, "config", "user.name", "RoleBench Tests")

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
            "task_source": "core",
            "recent_workspace": ".",
            "recent_days": 14,
            "recent_commits": 5,
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

    def test_recent_task_source_uses_workspace_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            changed = root / "src" / "main.py"
            changed.parent.mkdir()
            changed.write_text("print('recent')\n", encoding="utf-8")
            args = SimpleNamespace(
                tasks_file=None,
                task_source="recent",
                recent_workspace=str(root),
                recent_days=1,
                recent_commits=5,
                suite="core",
            )
            tasks = rolebench.load_tasks(args, ["task", "advisor"])
            self.assertEqual([task["role"] for task in tasks], ["task", "advisor"])
            self.assertIn("src/main.py", tasks[0]["prompt"])
            self.assertEqual(tasks[0]["workspace"], str(root.resolve()))
            self.assertEqual(tasks[0]["difficulty"]["level"], 3)
            self.assertIn("correctness", tasks[0]["dimension_targets"])
            self.assertEqual(args.recent_work["source"], "mtime")
            self.assertEqual(args.recent_work["workspace_count"], 1)
            self.assertEqual(rolebench.normalize_recent_workspaces(f"{root},{root}"), [str(root.resolve())])
            self.assertNotIn("git_error", args.recent_work)

    def test_task_schema_normalizes_evaluation_metadata_and_rejects_unsafe_paths(self):
        task = rolebench.normalize_task({
            "id": "paired-hard",
            "role": "task",
            "prompt": "make the bounded change",
            "pair_id": "paired",
            "variant": "hard",
            "difficulty": {"level": 3, "scope": 2, "reasoning": 3},
            "expected": {"output_contains": "verified"},
            "invariants": [{"type": "file_contains", "path": "src/result.txt", "value": "PASS"}],
        })
        self.assertEqual(task["schema_version"], 1)
        self.assertEqual(task["expected"]["output_contains"], ["verified"])
        self.assertEqual(task["invariants"][0]["dimensions"], ["correctness"])
        self.assertIn("correctness", task["dimension_targets"])
        with self.assertRaisesRegex(rolebench.RoleBenchError, "stay inside"):
            rolebench.normalize_task({
                "id": "unsafe",
                "role": "task",
                "prompt": "bad fixture",
                "invariants": [{"type": "file_exists", "path": "../secret"}],
            })
        with self.assertRaisesRegex(rolebench.RoleBenchError, "unknown dimension"):
            rolebench.normalize_task({"id": "bad-dimension", "role": "task", "prompt": "bad", "dimension_targets": ["style"]})

    def test_deterministic_evaluator_detects_out_of_scope_change(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            before = rolebench.snapshot_workspace(workspace)
            (workspace / "result.txt").write_text("PASS", encoding="utf-8")
            (workspace / "unrelated.txt").write_text("changed", encoding="utf-8")
            after = rolebench.snapshot_workspace(workspace)
            workspace_diff = rolebench.diff_workspace_snapshots(before, after)
            task = rolebench.normalize_task({
                "id": "task-eval",
                "role": "task",
                "prompt": "write result only",
                "dimension_targets": ["correctness", "instruction_following", "safety"],
                "invariants": [
                    {"type": "file_content_exact", "path": "result.txt", "value": "PASS"},
                    {"type": "allowed_changed_paths", "patterns": ["result.txt"]},
                ],
            })
            result = {
                "assignment_id": "a",
                "task_id": "task-eval",
                "role": "task",
                "model": "p/a",
                "thinking": None,
                "replicate": 1,
                "status": "pass",
                "output": "done",
            }
            evaluation = rolebench.build_evaluation(task, result, workspace, workspace_diff, False, None)
            self.assertEqual(evaluation["outcome"], "fail")
            self.assertEqual(evaluation["dimensions"]["correctness"]["status"], "pass")
            self.assertEqual(evaluation["dimensions"]["safety"]["status"], "fail")
            self.assertEqual(evaluation["hard_gate_status"], "fail")
            self.assertIn("unrelated.txt", workspace_diff["patch"])

    def test_recent_git_source_normalizes_subdir_and_special_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = root / "pkg"
            workspace.mkdir()
            (root / "other").mkdir()
            self.init_git_repo(root)
            (workspace / "tracked.py").write_text("print('tracked')\n", encoding="utf-8")
            (workspace / "deleted.png").write_bytes(b"deleted")
            self.git(root, "add", ".")
            self.git(root, "commit", "-qm", "initial")
            (workspace / "recent.py").write_text("print('recent')\n", encoding="utf-8")
            (root / "other" / "outside.py").write_text("print('outside')\n", encoding="utf-8")
            (workspace / "中文 shot.png").write_bytes(b"image")
            self.git(root, "add", ".")
            self.git(root, "commit", "-qm", "recent changes")
            (workspace / "deleted.png").unlink()
            self.git(root, "mv", "pkg/recent.py", "pkg/renamed file.py")
            untracked = workspace / "assets" / "新图.png"
            untracked.parent.mkdir()
            untracked.write_bytes(b"new image")

            recent = rolebench.collect_recent_work(str(workspace), 30, 5)

            self.assertEqual(recent["source"], "git")
            self.assertEqual(recent["workspace"], str(workspace.resolve()))
            self.assertIn("recent changes", {item["subject"] for item in recent["commits"]})
            self.assertIn("recent.py", recent["changed_files"])
            self.assertIn("renamed file.py", recent["changed_files"])
            self.assertIn("assets/新图.png", recent["changed_files"])
            self.assertIn("中文 shot.png", recent["changed_files"])
            self.assertNotIn("../other/outside.py", recent["changed_files"])
            self.assertNotIn("deleted.png", recent["images"])
            self.assertIn("中文 shot.png", recent["images"])
            self.assertIn("assets/新图.png", recent["images"])
            self.assertTrue(any("recent.py -> renamed file.py" in line for line in recent["working_tree"]))

            task = rolebench.build_recent_tasks(["vision"], recent)[0]
            self.assertEqual(task["workspace"], str(workspace.resolve()))
            self.assertEqual(task["image"], [str(workspace.resolve() / "中文 shot.png")])
            self.assertTrue(Path(task["image"][0]).is_file())

    def test_recent_git_error_is_recorded_and_shown_in_plan(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "recent.py").write_text("print('recent')\n", encoding="utf-8")
            with patch.object(rolebench, "run_git", return_value=(None, "git status failed: permission denied", 1)):
                recent = rolebench.collect_recent_work(str(root), 1, 5)
            self.assertEqual(recent["source"], "mtime")
            self.assertIn("permission denied", recent["git_error"])
            manifest = {
                "candidates": [{"selector": "p/a"}],
                "roles": ["task"],
                "mode": "all-same",
                "executor": "direct",
                "task_source": "recent",
                "tasks": [{"id": "recent-task", "role": "task", "prompt": "inspect"}],
                "runs": 1,
                "estimate": {"assignments": 1, "calls": 1, "input_tokens": 1, "output_tokens": 1, "cost_usd": None, "cost_status": "unknown"},
                "recent_work": {"source": "mtime", "workspace_count": 1, "workspaces": [recent]},
            }
            plan = rolebench.format_plan(manifest)
            self.assertIn(f"{root.resolve()} [mtime]", plan)
            self.assertIn("recent work warnings", plan)
            self.assertIn("permission denied", plan)

    @patch("rolebench.subprocess.run")
    def test_run_git_returns_decoded_error_details(self, run):
        run.return_value = SimpleNamespace(returncode=1, stdout=b"", stderr=b"fatal: permission denied\xff")
        output, error, returncode = rolebench.run_git(Path("."), ["status"])
        self.assertIsNone(output)
        self.assertEqual(returncode, 1)
        self.assertIn("permission denied", error)

    def test_recent_task_source_reports_invalid_or_empty_workspace(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(rolebench.RoleBenchError):
                rolebench.collect_recent_work(str(root / "missing"), 1, 5)
            with self.assertRaises(rolebench.RoleBenchError):
                rolebench.collect_recent_work(str(root), 1, 5)
            with patch.object(rolebench, "run_git", return_value=(None, "git log failed: permission denied", 1)):
                with self.assertRaisesRegex(rolebench.RoleBenchError, "git error: git log failed"):
                    rolebench.collect_recent_work(str(root), 1, 5)
            args = SimpleNamespace(
                tasks_file=None,
                task_source="recent",
                recent_workspace=str(root),
                recent_days=1,
                recent_commits=5,
                suite="core",
            )
            with self.assertRaisesRegex(rolebench.RoleBenchError, "requires selected roles"):
                rolebench.load_tasks(args)

    def test_recent_run_writes_manifest_and_plan_details(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "src").mkdir()
            (root / "src" / "main.py").write_text("print('recent')\n", encoding="utf-8")
            output = root / "run"
            args = self.make_run_args(
                root,
                mode="all-same",
                task_source="recent",
                recent_workspace=[str(root)],
                recent_days=1,
                recent_commits=5,
                execute=False,
                confirm=False,
                output=str(output),
            )
            self.assertEqual(rolebench.run_command(args), 0)
            manifest = rolebench.load_json_document(output / "manifest.json")
            self.assertEqual(manifest["task_source"], "recent")
            self.assertEqual(manifest["recent_work"]["workspace_count"], 1)
            self.assertIn(f"{root.resolve()} [mtime]", rolebench.format_plan(manifest))

    def test_main_rejects_non_positive_recent_window(self):
        with tempfile.TemporaryDirectory() as directory:
            output = str(Path(directory) / "run")
            for option in ("--recent-days", "--recent-commits"):
                self.assertEqual(
                    rolebench.main(["run", "--models", "p/a", "--roles", "task", "--mode", "all-same", option, "0", "--output", output]),
                    2,
                )

    def test_recent_task_source_supports_multiple_workspaces(self):
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            roots = [Path(first), Path(second)]
            for index, root in enumerate(roots, 1):
                changed = root / "src" / f"main{index}.py"
                changed.parent.mkdir()
                changed.write_text(f"print({index})\n", encoding="utf-8")
            args = SimpleNamespace(
                tasks_file=None,
                task_source="recent",
                recent_workspace=[str(roots[0]), str(roots[1])],
                recent_days=1,
                recent_commits=5,
                suite="core",
            )
            tasks = rolebench.load_tasks(args, ["task", "advisor"])
            self.assertEqual(len(tasks), 4)
            self.assertEqual({task["role"] for task in tasks}, {"task", "advisor"})
            self.assertEqual({task["workspace"] for task in tasks}, {str(root.resolve()) for root in roots})
            self.assertEqual(len({task["id"] for task in tasks}), 4)
            self.assertTrue(all("src/main1.py" in task["prompt"] and "src/main2.py" not in task["prompt"] for task in tasks[:2]))
            self.assertTrue(all("src/main2.py" in task["prompt"] and "src/main1.py" not in task["prompt"] for task in tasks[2:]))
            self.assertEqual(args.recent_work["workspace_count"], 2)
            self.assertEqual(len(args.recent_work["workspaces"]), 2)

    def test_recent_workspaces_reports_mixed_sources(self):
        with tempfile.TemporaryDirectory() as git_directory, tempfile.TemporaryDirectory() as plain_directory:
            git_root = Path(git_directory)
            plain_root = Path(plain_directory)
            self.init_git_repo(git_root)
            (git_root / "tracked.py").write_text("print('git')\n", encoding="utf-8")
            self.git(git_root, "add", ".")
            self.git(git_root, "commit", "-qm", "git work")
            (plain_root / "plain.py").write_text("print('mtime')\n", encoding="utf-8")
            recent = rolebench.collect_recent_workspaces([str(git_root), str(plain_root)], 1, 5)
            self.assertEqual(recent["source"], "mixed")
            self.assertEqual([item["source"] for item in recent["workspaces"]], ["git", "mtime"])

    def test_deleted_image_does_not_become_vision_fixture(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.init_git_repo(root)
            image = root / "deleted.png"
            image.write_bytes(b"image")
            self.git(root, "add", ".")
            self.git(root, "commit", "-qm", "add image")
            image.unlink()
            recent = rolebench.collect_recent_work(str(root), 1, 5)
            self.assertIn("deleted.png", recent["changed_files"])
            self.assertEqual(recent["images"], [])
            task = rolebench.build_recent_tasks(["vision"], recent)[0]
            self.assertNotIn("requires_image", task)
            self.assertNotIn("image", task)

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
        self.assertEqual(row["_evaluation"]["execution_status"], "timeout")
        self.assertEqual(row["_evaluation"]["outcome"], "fail")
        self.assertEqual(row["_evaluation"]["hard_gate_status"], "unknown")
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
            evaluations = [
                {"role": "task", "model": "p/a", "thinking": "high", "outcome": "unknown", "dimensions": {}},
                {"role": "task", "model": "p/a", "thinking": "low", "outcome": "unknown", "dimensions": {}},
            ]
            (root / "evaluations.jsonl").write_text("\n".join(json.dumps(row) for row in evaluations) + "\n", encoding="utf-8")
            output = root / "report.json"
            self.assertEqual(rolebench.main(["report", "--run", str(root), "--format", "json", "--output", str(output)]), 0)
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertIn("task/p/a:high", payload["groups"])
            self.assertIn("task/p/a:low", payload["groups"])
            self.assertIn("task/p/a:high", payload["evaluation_groups"])
            self.assertIn("task/p/a:low", payload["evaluation_groups"])

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

    def test_report_includes_outcomes_dimensions_and_failed_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "manifest.json").write_text(json.dumps({"generated_at": "now", "mode": "role-isolated", "roles": ["task"], "assignments": []}), encoding="utf-8")
            (root / "results.jsonl").write_text(json.dumps({"role": "task", "model": "p/a", "status": "pass", "latency_ms": 1, "output": "done"}) + "\n", encoding="utf-8")
            evaluation = {
                "role": "task", "model": "p/a", "thinking": None, "task_id": "task-1", "outcome": "fail", "hard_gate_status": "fail",
                "checks": [{"id": "scope", "status": "fail", "evidence": "changed forbidden.txt"}],
                "dimensions": {"safety": {"status": "fail", "score": 0}},
            }
            (root / "evaluations.jsonl").write_text(json.dumps(evaluation) + "\n", encoding="utf-8")
            output = root / "report.md"
            self.assertEqual(rolebench.main(["report", "--run", str(root), "--output", str(output)]), 0)
            report = output.read_text(encoding="utf-8")
            self.assertIn("## Outcomes", report)
            self.assertIn("## Dimensions", report)
            self.assertIn("changed forbidden.txt", report)

    def test_fake_executor_end_to_end(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "config.yml"
            config.write_text("modelRoles:\n  task: fake/model\n", encoding="utf-8")
            tasks = root / "tasks.json"
            tasks.write_text(json.dumps({"schema_version": 1, "tasks": [{
                "id": "task-1", "role": "task", "prompt": "do", "workspace": str(root), "expected": {"output_exact": "PASS"},
                "dimension_targets": ["correctness", "instruction_following"],
            }]}), encoding="utf-8")
            output = root / "run"
            fake_omp = (Path(__file__).parent / "fake_omp").resolve()
            self.assertEqual(rolebench.main([
                "run", "--omp-bin", str(fake_omp), "--models", "fake/model", "--roles", "task",
                "--config-file", str(config), "--tasks-file", str(tasks), "--execute", "--confirm", "--output", str(output),
            ]), 0)
            result = json.loads((output / "results.jsonl").read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(result["status"], "pass")
            self.assertEqual(result["execution_status"], "completed")
            self.assertNotIn("outcome", result)
            evaluation = json.loads((output / "evaluations.jsonl").read_text(encoding="utf-8").splitlines()[0])
            self.assertEqual(evaluation["outcome"], "pass")
            self.assertEqual(evaluation["dimensions"]["correctness"]["score"], 2)
            for artifact in evaluation["artifacts"].values():
                self.assertTrue((output / artifact).is_file())
            before_artifact = json.loads((output / evaluation["artifacts"]["workspace_before"]).read_text(encoding="utf-8"))
            self.assertFalse(any(path.startswith("run/") for path in before_artifact["files"]))

    def test_command_checks_require_explicit_trust_and_run_after_approval(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "config.yml"
            config.write_text("modelRoles:\n  task: fake/model\n", encoding="utf-8")
            tasks = root / "tasks.json"
            tasks.write_text(json.dumps([{
                "id": "task-command", "role": "task", "prompt": "do", "expected": "PASS",
                "invariants": [{"type": "command", "argv": [sys.executable, "-c", "print('CHECKED')"], "stdout_contains": "CHECKED"}],
            }]), encoding="utf-8")
            fake_omp = (Path(__file__).parent / "fake_omp").resolve()
            blocked = root / "blocked"
            common = [
                "run", "--omp-bin", str(fake_omp), "--models", "fake/model", "--roles", "task",
                "--config-file", str(config), "--tasks-file", str(tasks), "--execute", "--confirm",
            ]
            self.assertEqual(rolebench.main(common + ["--output", str(blocked)]), 2)
            manifest = json.loads((blocked / "manifest.json").read_text(encoding="utf-8"))
            self.assertIn(sys.executable, rolebench.format_plan(manifest))
            self.assertEqual(json.loads((blocked / "summary.json").read_text(encoding="utf-8"))["reason"], "command_checks_not_approved")
            self.assertFalse((blocked / "results.jsonl").exists())
            allowed = root / "allowed"
            self.assertEqual(rolebench.main(common + ["--allow-check-command", "--output", str(allowed)]), 0)
            evaluation = json.loads((allowed / "evaluations.jsonl").read_text(encoding="utf-8").splitlines()[0])
            command_check = next(check for check in evaluation["checks"] if check["type"] == "command")
            self.assertEqual(command_check["status"], "pass")
            self.assertEqual(evaluation["outcome"], "pass")

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
