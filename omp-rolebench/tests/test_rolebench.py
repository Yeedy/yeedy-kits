import json
import os
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

    def test_fixed_judge_prompt_is_blind_and_versioned(self):
        task = {
            "id": "task-judge",
            "role": "advisor",
            "prompt": "Assess the evidence.",
            "workspace": "/sensitive/workspace",
            "image": "/sensitive/image.png",
            "dimension_targets": ["evidence_quality"],
        }
        result = {"model": "secret/provider", "execution_status": "completed", "output": "The result is supported."}
        deterministic = {"execution_status": "completed", "outcome": "pass", "hard_gate_status": "pass", "checks": []}
        prompt = rolebench.build_fixed_judge_prompt(task, result, deterministic)
        self.assertIn(rolebench.FIXED_JUDGE_RUBRIC_VERSION, prompt)
        self.assertIn("evidence_quality", prompt)
        self.assertNotIn("secret/provider", prompt)
        self.assertNotIn("/sensitive/workspace", prompt)
        self.assertNotIn("/sensitive/image.png", prompt)

    def test_fixed_judge_estimate_includes_repeated_calls_and_cost(self):
        pricing = {"judge/model": {"input_per_million": 1.0, "output_per_million": 2.0}}
        assignments = [{"id": "assignment-001", "mode": "all-same", "mapping": {"task": "p/a"}}]
        tasks = [{"id": "task-1", "role": "task", "prompt": "do", "expected": {"output_contains": ["PASS"]}}]
        estimate = rolebench.estimate_fixed_judge(["task"], assignments, tasks, 3, "judge/model", pricing)
        self.assertTrue(estimate["enabled"])
        self.assertEqual(estimate["calls"], 3)
        self.assertEqual(estimate["output_tokens"], 3 * rolebench.FIXED_JUDGE_OUTPUT_TOKENS)
        self.assertEqual(estimate["cost_status"], "estimated")
        self.assertGreater(estimate["cost_usd"], 0)

    def test_fixed_judge_evaluation_normalizes_scores_and_missing_dimensions(self):
        task = {"id": "task-judge", "role": "task", "dimension_targets": ["correctness", "evidence_quality", "safety"]}
        result = {"assignment_id": "a", "task_id": "task-judge", "role": "task", "model": "p/a", "replicate": 1, "execution_status": "completed"}
        judge_result = {"judge_result_id": "a:task-judge:1"}
        evaluation = rolebench.build_judge_evaluation(
            task,
            result,
            judge_result,
            {
                "correctness": {"status": "pass", "score": 0, "evidence": "matched fixture"},
                "evidence_quality": {"status": "unknown", "score": 2, "evidence": ["not enough evidence"]},
            },
        )
        self.assertEqual(evaluation["dimensions"]["correctness"]["score"], 2)
        self.assertEqual(evaluation["dimensions"]["evidence_quality"]["score"], None)
        self.assertEqual(evaluation["dimensions"]["safety"]["status"], "unknown")
        self.assertEqual(evaluation["hard_gate_status"], "unknown")
        self.assertEqual(evaluation["outcome"], "unknown")

    @patch("rolebench.subprocess.run")
    def test_fixed_judge_keeps_raw_result_separate_from_evaluation(self, run):
        judge_payload = {
            "dimensions": {"correctness": {"status": "pass", "score": 2, "evidence": ["output matched"]}},
            "summary": "matched the declared contract",
        }
        run.return_value = SimpleNamespace(returncode=0, stdout=json.dumps({"text": json.dumps(judge_payload)}), stderr="")
        task = {"id": "task-judge", "role": "task", "prompt": "do", "dimension_targets": ["correctness"]}
        result = {"assignment_id": "a", "task_id": "task-judge", "role": "task", "model": "secret/provider", "replicate": 1, "execution_status": "completed", "output": "PASS"}
        deterministic = {"execution_status": "completed", "outcome": "pass", "hard_gate_status": "pass", "checks": []}
        raw, evaluation = rolebench.run_fixed_judge("omp", "judge/model:high", task, result, deterministic, timeout=5)
        self.assertEqual(raw["model"], "judge/model")
        self.assertEqual(raw["thinking"], "high")
        self.assertEqual(raw["status"], "pass")
        self.assertEqual(evaluation["evaluator"], "fixed_judge")
        self.assertEqual(evaluation["dimensions"]["correctness"]["score"], 2)
        prompt = run.call_args.args[0][-1]
        self.assertNotIn("secret/provider", prompt)
        command = run.call_args.args[0]
        self.assertIn("judge/model", command)
        self.assertIn("--thinking", command)
        self.assertIn("high", command)

    def test_recommendations_require_repeated_hard_gate_evidence(self):
        task = {"id": "task-1", "role": "task", "dimension_targets": ["correctness", "instruction_following"]}
        results = []
        evaluations = []
        judge_results = []
        judge_evaluations = []
        for replicate in range(1, 4):
            identity = {"assignment_id": "assignment-001", "task_id": "task-1", "replicate": replicate}
            results.append({**identity, "role": "task", "model": "p/a", "status": "pass", "latency_ms": replicate, "usage": {"cost": {"total": 0.001}}})
            evaluations.append({**identity, "role": "task", "model": "p/a", "outcome": "pass", "hard_gate_status": "pass", "dimensions": {"correctness": {"status": "pass", "score": 2, "hard_gate": True}}})
            judge_results.append({**identity, "model": "judge/model", "usage": {"cost": {"total": 0.0005}}})
            judge_evaluations.append({**identity, "role": "task", "model": "p/a", "outcome": "pass", "hard_gate_status": "pass", "dimensions": {"correctness": {"status": "pass", "score": 2, "hard_gate": True}, "instruction_following": {"status": "pass", "score": 2, "hard_gate": False}}})
        recommendation = rolebench.build_recommendations(["task"], [task], results, evaluations, judge_results, judge_evaluations, 3)
        self.assertEqual(recommendation["status"], "complete")
        self.assertEqual(recommendation["roles"][0]["status"], "recommended")
        self.assertEqual(recommendation["roles"][0]["model"], "p/a")
        unknown_cost = rolebench.build_recommendations(
            ["task"],
            [task],
            [{key: value for key, value in row.items() if key != "usage"} for row in results],
            evaluations,
            [{key: value for key, value in row.items() if key != "usage"} for row in judge_results],
            judge_evaluations,
            3,
        )
        self.assertEqual(unknown_cost["roles"][0]["status"], "recommended")
        self.assertIsNone(unknown_cost["candidates"][0]["total_cost_usd"])
        self.assertEqual(unknown_cost["candidates"][0]["cost_status"], "unknown")

        blocked = rolebench.build_recommendations(["task"], [task], results[:1], evaluations[:1], judge_results[:1], judge_evaluations[:1], 3)
        self.assertEqual(blocked["status"], "blocked")
        self.assertIn("at least 3 runs", blocked["candidates"][0]["blocked_reasons"][0])

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

    def test_task_schema_rejects_unknown_keys_nul_commands_and_oversized_regex(self):
        base = {"id": "bad", "role": "task", "prompt": "do"}
        with self.assertRaisesRegex(rolebench.RoleBenchError, "unknown top-level key.*invarients"):
            rolebench.normalize_task({**base, "invarients": []})
        with self.assertRaisesRegex(rolebench.RoleBenchError, "NUL"):
            rolebench.normalize_task({**base, "invariants": [{"type": "command", "argv": ["bad\0argv"]}]})
        with self.assertRaisesRegex(rolebench.RoleBenchError, "exceeds"):
            rolebench.normalize_task({**base, "expected": {"output_regex": "a" * (rolebench.REGEX_PATTERN_LIMIT + 1)}})

    def test_expected_checks_cover_contracts_and_incomplete_execution(self):
        output = '{"ok": 1}'
        task = rolebench.normalize_task({
            "id": "output-contracts",
            "role": "task",
            "prompt": "respond",
            "expected": {
                "output_exact": output,
                "output_contains": ["ok"],
                "output_regex": [r'"ok"\s*:\s*1'],
                "output_not_regex": ["secret"],
                "output_json_equals": {"ok": 1},
                "max_output_chars": len(output),
            },
        })
        checks = rolebench.expected_output_checks(task, output)
        self.assertEqual({check["type"] for check in checks}, set(rolebench.EXPECTED_KEYS))
        self.assertTrue(all(check["status"] == "pass" for check in checks))

        result = {
            "assignment_id": "a",
            "task_id": task["id"],
            "role": "task",
            "model": "p/a",
            "thinking": None,
            "replicate": 1,
            "status": "error",
            "error_type": "timeout",
            "output": "",
        }
        evaluation = rolebench.build_evaluation(task, result, None, None, False, None)
        self.assertTrue(all(check["status"] == "unknown" for check in evaluation["checks"]))
        self.assertIsNone(evaluation["dimensions"]["correctness"]["score"])
        self.assertEqual(evaluation["hard_gate_status"], "unknown")

    def test_regex_and_json_resource_limits_become_unknown(self):
        matched, error = rolebench.bounded_regex_search("(a+)+$", "a" * 40 + "!")
        self.assertIsNone(matched)
        self.assertIn("exceeded", error)
        matched, error = rolebench.bounded_regex_search("a", "a" * (rolebench.REGEX_INPUT_LIMIT + 1))
        self.assertIsNone(matched)
        self.assertIn("input exceeds", error)

        task = rolebench.normalize_task({
            "id": "deep-json",
            "role": "task",
            "prompt": "respond",
            "expected": {"output_json_equals": []},
        })
        result = {
            "assignment_id": "a",
            "task_id": task["id"],
            "role": "task",
            "model": "p/a",
            "thinking": None,
            "replicate": 1,
            "status": "pass",
            "output": "[" * 5000,
        }
        evaluation = rolebench.build_evaluation(task, result, None, None, False, None)
        self.assertEqual(evaluation["checks"][0]["status"], "unknown")
        self.assertEqual(evaluation["outcome"], "unknown")

    def test_failure_mode_patterns_cover_handling_and_honesty(self):
        task = rolebench.normalize_task({
            "id": "failure-contract",
            "role": "task",
            "prompt": "explain the failure",
            "failure_mode": {
                "required_output_regex": ["cannot proceed"],
                "forbidden_output_regex": ["completed successfully"],
            },
        })
        checks = rolebench.deterministic_checks(task, {"output": "cannot proceed; completed successfully"}, None, None, False)
        self.assertEqual([check["status"] for check in checks], ["pass", "fail"])
        self.assertEqual(checks[1]["dimensions"], ["honesty"])
        self.assertTrue(checks[1]["hard_gate"])

    def test_workspace_snapshot_handles_special_git_and_unavailable_text(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            (workspace / ".git" / "hooks").mkdir(parents=True)
            hook = workspace / ".git" / "hooks" / "pre-commit"
            hook.write_text("first", encoding="utf-8")
            gone = workspace / "gone.txt"
            gone.write_text("bye", encoding="utf-8")
            fifo = workspace / "events.fifo"
            os.mkfifo(fifo)
            with patch.object(rolebench, "SNAPSHOT_TEXT_LIMIT", 3):
                large_text = workspace / "large.txt"
                large_text.write_text("old!", encoding="utf-8")
                before = rolebench.snapshot_workspace(workspace)
                hook.write_text("second", encoding="utf-8")
                large_text.write_text("new!", encoding="utf-8")
                gone.unlink()
                after = rolebench.snapshot_workspace(workspace)
            workspace_diff = rolebench.diff_workspace_snapshots(before, after)

            self.assertEqual(before["files"]["events.fifo"]["type"], "fifo")
            self.assertNotIn("text", before["files"][".git/hooks/pre-commit"])
            self.assertIn(".git/hooks/pre-commit", workspace_diff["changed_paths"])
            self.assertIn("large.txt", workspace_diff["text_unavailable"])
            self.assertNotIn("old!", workspace_diff["patch"])
            self.assertIn("gone.txt", workspace_diff["patch"])

            task = rolebench.normalize_task({
                "id": "protect-hooks",
                "role": "task",
                "prompt": "do",
                "invariants": [{"type": "forbidden_changed_paths", "patterns": [".git/hooks/**"]}],
            })
            checks = rolebench.deterministic_checks(task, {"output": "ok"}, workspace, workspace_diff, False)
            self.assertEqual(checks[0]["status"], "fail")

    def test_file_invariants_cover_content_absence_json_and_size_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            (workspace / "exact.txt").write_text("PASS", encoding="utf-8")
            (workspace / "contains.txt").write_text("alpha beta", encoding="utf-8")
            (workspace / "data.json").write_text('{"ok": true}', encoding="utf-8")
            (workspace / "dangling").symlink_to("missing-target")
            invariants = [
                {"type": "file_exists", "path": "exact.txt"},
                {"type": "file_content_exact", "path": "exact.txt", "value": "PASS"},
                {"type": "file_contains", "path": "contains.txt", "values": ["alpha", "beta"]},
                {"type": "file_matches", "path": "contains.txt", "pattern": r"alpha\s+beta"},
                {"type": "json_file_equals", "path": "data.json", "value": {"ok": True}},
                {"type": "file_not_exists", "path": "absent.txt"},
                {"type": "file_not_exists", "path": "dangling"},
            ]
            task = rolebench.normalize_task({"id": "files", "role": "task", "prompt": "do", "invariants": invariants})
            checks = rolebench.deterministic_checks(task, {"output": "ok"}, workspace, None, False)
            statuses = [check["status"] for check in checks]
            self.assertEqual(statuses[:-1], ["pass"] * 6)
            self.assertEqual(statuses[-1], "fail")

            (workspace / "large.txt").write_text("0123456789", encoding="utf-8")
            invariant = rolebench.normalize_invariant({"type": "file_contains", "path": "large.txt", "value": "0"}, "large", 1)
            with patch.object(rolebench, "FILE_CHECK_TEXT_LIMIT", 5):
                status, evidence = rolebench.file_invariant_check(invariant, workspace)
            self.assertEqual(status, "unknown")
            self.assertIn("limit", evidence)

    def test_path_globs_are_segment_aware_and_support_double_star(self):
        self.assertTrue(rolebench.path_glob_matches("root.txt", "*.txt"))
        self.assertFalse(rolebench.path_glob_matches("nested/root.txt", "*.txt"))
        self.assertTrue(rolebench.path_glob_matches("nested/root.txt", "**/*.txt"))
        self.assertTrue(rolebench.path_glob_matches(".git/hooks/pre-commit", ".git/hooks/**"))

    def test_baseline_symlink_escape_is_unknown_but_new_escape_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            link = workspace / "outside"
            link.symlink_to(Path(directory).parent)
            before = rolebench.snapshot_workspace(workspace)
            invariant = rolebench.normalize_invariant({"type": "file_exists", "path": "outside/secret"}, "link", 1)
            task = {"dimension_targets": ["correctness"], "invariants": [invariant]}
            unknown = rolebench.deterministic_checks(task, {"output": "ok"}, workspace, None, False, True, before)
            failed = rolebench.deterministic_checks(task, {"output": "ok"}, workspace, None, False, True, {"files": {}})
            self.assertEqual(unknown[0]["status"], "unknown")
            self.assertEqual(failed[0]["status"], "fail")

    def test_command_checks_limit_environment_output_and_runtime_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            env_invariant = rolebench.normalize_invariant({
                "type": "command",
                "argv": [sys.executable, "-c", "import os; print(os.getenv('ROLEBENCH_SECRET', 'missing'))"],
                "stdout_contains": "missing",
            }, "command", 1)
            with patch.dict(os.environ, {"ROLEBENCH_SECRET": "sensitive"}):
                status, _ = rolebench.run_command_invariant(env_invariant, workspace, True)
            self.assertEqual(status, "pass")

            output_invariant = rolebench.normalize_invariant({
                "type": "command",
                "argv": [sys.executable, "-c", "print('A' * 100)"],
                "stdout_contains": "Z",
            }, "command", 2)
            with patch.object(rolebench, "COMMAND_OUTPUT_LIMIT", 10):
                status, evidence = rolebench.run_command_invariant(output_invariant, workspace, True)
            self.assertEqual(status, "unknown")
            self.assertIn("truncated", evidence)

            invalid = {"argv": ["bad\0argv"], "timeout": 1, "expected_exit_code": 0}
            status, evidence = rolebench.run_command_invariant(invalid, workspace, True)
            self.assertEqual(status, "fail")
            self.assertIn("cannot run", evidence)

            timeout_invariant = rolebench.normalize_invariant({
                "type": "command",
                "argv": [sys.executable, "-c", "import time; time.sleep(2)"],
                "timeout": 1,
            }, "command", 3)
            status, evidence = rolebench.run_command_invariant(timeout_invariant, workspace, True)
            self.assertEqual(status, "fail")
            self.assertIn("timed out", evidence)

    def test_zero_hard_gate_evidence_cannot_pass(self):
        task = rolebench.normalize_task({
            "id": "unsupported",
            "role": "task",
            "prompt": "do",
            "dimension_targets": ["evidence_quality"],
        })
        result = {
            "assignment_id": "a",
            "task_id": task["id"],
            "role": "task",
            "model": "p/a",
            "thinking": None,
            "replicate": 1,
            "status": "pass",
            "output": "done",
        }
        evaluation = rolebench.build_evaluation(task, result, None, None, False, None)
        self.assertEqual(evaluation["hard_gate_status"], "unknown")
        self.assertEqual(evaluation["outcome"], "unknown")
        self.assertIsNone(evaluation["dimensions"]["evidence_quality"]["score"])

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

    @patch("rolebench.write_workspace_artifacts", side_effect=OSError("disk full"))
    @patch("rolebench.subprocess.run")
    def test_artifact_failure_does_not_overwrite_successful_execution(self, run, _write_artifacts):
        run.return_value = SimpleNamespace(returncode=0, stdout=json.dumps({"text": "PASS"}), stderr="")
        assignment = {"id": "a", "mode": "role-isolated", "candidate": "p/a", "mapping": {"task": "p/a"}}
        task = {"id": "task-001", "role": "task", "prompt": "do it"}
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            row = rolebench.run_one("omp", assignment, task, 1, output, 5, output, "direct", None, artifacts_root=output / "artifacts")
        self.assertEqual(row["status"], "pass")
        self.assertEqual(row["execution_status"], "completed")
        self.assertNotIn("error_type", row)
        self.assertEqual(row["warnings"][0]["type"], "artifact_write")
        self.assertEqual(row["_evaluation"]["evaluation_status"], "completed")

    @patch("rolebench.build_evaluation", side_effect=RuntimeError("bad evaluator"))
    @patch("rolebench.subprocess.run")
    def test_evaluator_failure_preserves_raw_result_and_returns_unknown_evaluation(self, run, _build_evaluation):
        run.return_value = SimpleNamespace(returncode=0, stdout=json.dumps({"text": "PASS"}), stderr="")
        assignment = {"id": "a", "mode": "role-isolated", "candidate": "p/a", "mapping": {"task": "p/a"}}
        task = {"id": "task-001", "role": "task", "prompt": "do it", "dimension_targets": ["correctness"]}
        row = rolebench.run_one("omp", assignment, task, 1, Path("."), 5, Path("."), "direct", None)
        self.assertEqual(row["status"], "pass")
        self.assertEqual(row["execution_status"], "completed")
        self.assertEqual(row["warnings"][0]["type"], "evaluation")
        self.assertEqual(row["_evaluation"]["evaluation_status"], "failed")
        self.assertEqual(row["_evaluation"]["outcome"], "unknown")

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

    def test_cell_internal_error_is_recorded_and_later_cells_continue(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = self.make_run_args(root, runs=2)
            second = {
                "assignment_id": "role-task-p-a",
                "mode": "role-isolated",
                "role": "task",
                "task_id": "task-1",
                "replicate": 2,
                "model": "p/a",
                "provider_call_started": False,
                "status": "pass",
                "execution_status": "completed",
                "output": "second",
            }
            task = {"id": "task-1", "role": "task", "prompt": "do", "dimension_targets": ["correctness"]}
            with patch.object(rolebench, "load_tasks", return_value=[task]):
                with patch.object(rolebench, "run_one", side_effect=[RuntimeError("broken cell"), second]) as run:
                    self.assertEqual(rolebench.run_command(args), 0)
            rows = [json.loads(line) for line in (root / "run" / "results.jsonl").read_text(encoding="utf-8").splitlines()]
            summary = json.loads((root / "run" / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(run.call_count, 2)
            self.assertEqual(rows[0]["error_type"], "harness_internal")
            self.assertEqual(rows[1]["output"], "second")
            self.assertEqual(summary["result_rows"], 2)
            self.assertEqual(summary["evaluation_rows"], 1)

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

    def test_fixed_judge_cost_is_included_in_cap_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pricing = root / "pricing.json"
            pricing.write_text(json.dumps({
                "p/a": {"input_per_million": 100.0, "output_per_million": 100.0},
                "judge/model": {"input_per_million": 100.0, "output_per_million": 100.0},
            }), encoding="utf-8")
            args = self.make_run_args(root, pricing_file=str(pricing), max_cost_usd=0.001, judge_model="judge/model", recommend=True)
            task = {"id": "task-1", "role": "task", "prompt": "do", "dimension_targets": ["correctness"]}
            with patch.object(rolebench, "load_tasks", return_value=[task]):
                with patch.object(rolebench, "run_one") as run:
                    self.assertEqual(rolebench.run_command(args), 2)
            run.assert_not_called()
            summary = json.loads((root / "run" / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["reason"], "estimated_cost_exceeds_cap")
            self.assertGreater(summary["model_cost_usd"], 0)
            self.assertGreater(summary["judge_cost_usd"], 0)
            self.assertEqual(summary["estimated_cost_usd"], summary["model_cost_usd"] + summary["judge_cost_usd"])

    def test_recommendation_preview_does_not_write_unexecuted_result(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = self.make_run_args(root, execute=False, confirm=False, judge_model="judge/model", recommend=True)
            with patch.object(rolebench, "load_tasks", return_value=[{"id": "task-1", "role": "task", "prompt": "do", "dimension_targets": ["correctness"]}]):
                self.assertEqual(rolebench.run_command(args), 0)
            self.assertFalse((root / "run" / "recommendations.json").exists())
            summary = json.loads((root / "run" / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["recommendations"], "not-executed")

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

    def test_stale_fixed_judge_results_are_rotated_without_judge(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "config.yml"
            config.write_text("modelRoles:\n  task: p/base\n", encoding="utf-8")
            output = root / "run"
            output.mkdir()
            old_judge = output / "judge_results.jsonl"
            old_evaluation = output / "judge_evaluations.jsonl"
            old_recommendation = output / "recommendations.json"
            old_judge.write_text('{"model":"judge/old"}\n', encoding="utf-8")
            old_evaluation.write_text('{"evaluator":"fixed_judge"}\n', encoding="utf-8")
            old_recommendation.write_text('{"status":"complete"}\n', encoding="utf-8")
            self.assertEqual(rolebench.main(["run", "--models", "p/new", "--roles", "task", "--config-file", str(config), "--output", str(output)]), 0)
            self.assertFalse(old_judge.exists())
            self.assertFalse(old_evaluation.exists())
            self.assertFalse(old_recommendation.exists())
            self.assertEqual(len(list(output.glob("judge_results.jsonl.previous-*"))), 1)
            self.assertEqual(len(list(output.glob("judge_evaluations.jsonl.previous-*"))), 1)
            self.assertEqual(len(list(output.glob("recommendations.json.previous-*"))), 1)

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

    def test_report_includes_fixed_judge_and_recommendation_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "manifest.json").write_text(json.dumps({"generated_at": "now", "mode": "all-same", "roles": ["task"], "assignments": [], "evaluation": {"fixed_judge": {"enabled": True, "rubric_version": "rolebench-v2-test"}}}), encoding="utf-8")
            (root / "judge_evaluations.jsonl").write_text(json.dumps({"role": "task", "task_id": "task-1", "model": "p/a", "outcome": "pass", "hard_gate_status": "pass", "rubric_version": "rolebench-v2-test", "summary": "supported"}) + "\n", encoding="utf-8")
            (root / "recommendations.json").write_text(json.dumps({"status": "complete", "roles": [{"role": "task", "model": "p/a", "status": "recommended", "reason": "hard gates passed"}]}), encoding="utf-8")
            output = root / "report.md"
            self.assertEqual(rolebench.main(["report", "--run", str(root), "--output", str(output)]), 0)
            report = output.read_text(encoding="utf-8")
            self.assertIn("## Fixed Judge", report)
            self.assertIn("## Recommendations", report)
            self.assertIn("rolebench-v2-test", report)
            json_output = root / "report.json"
            self.assertEqual(rolebench.main(["report", "--run", str(root), "--format", "json", "--output", str(json_output)]), 0)
            payload = json.loads(json_output.read_text(encoding="utf-8"))
            self.assertEqual(payload["recommendations"]["status"], "complete")
            self.assertEqual(len(payload["judge_evaluations"]), 1)

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
