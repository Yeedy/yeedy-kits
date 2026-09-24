import json
import math
import os
import subprocess
import sys
import tempfile
import threading
import time
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
            "jobs": 1,
            "execute": True,
            "resume": False,
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
        self.assertEqual(rolebench.extract_cost({"input": 2, "output": 3, "cost": {"input": 0.004, "output": 0.008, "total": 0.012}}), 0.012)
        self.assertIsNone(rolebench.extract_cost({"input": 2, "output": 3, "total": 5}))

    def test_omp_event_stream_uses_assistant_output_and_aggregates_usage(self):
        payload = [
            {
                "type": "message_end",
                "message": {
                    "role": "user",
                    "content": [{"type": "text", "text": "PROMPT"}],
                    "usage": {"input": 0, "output": 0, "totalTokens": 0, "cost": {"total": 0}},
                },
            },
            {
                "type": "message_end",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "toolCall", "name": "read"}],
                    "usage": {"input": 10, "output": 2, "totalTokens": 12, "cost": {"input": 0.1, "output": 0.02, "total": 0.12}},
                },
            },
            {
                "type": "message_end",
                "message": {"role": "toolResult", "content": [{"type": "text", "text": "TOOL OUTPUT"}]},
            },
            {
                "type": "message_end",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "thinking", "thinking": "private"}, {"type": "text", "text": "FINAL"}],
                    "usage": {"input": 20, "output": 4, "reasoningTokens": 1, "totalTokens": 24, "cost": {"input": 0.2, "output": 0.04, "total": 0.24}},
                },
            },
        ]
        self.assertEqual(rolebench.extract_output(payload), "FINAL")
        usage = rolebench.extract_usage(payload)
        self.assertEqual(usage["input"], 30)
        self.assertEqual(usage["output"], 6)
        self.assertEqual(usage["reasoningTokens"], 1)
        self.assertAlmostEqual(rolebench.extract_cost(usage), 0.36)
        payload[-1]["message"].pop("usage")
        self.assertIsNone(rolebench.extract_usage(payload))

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
        self.assertIn("do not inspect a workspace or attempt any tool calls", prompt)
        self.assertNotIn("secret/provider", prompt)
        self.assertNotIn("/sensitive/workspace", prompt)
        self.assertNotIn("/sensitive/image.png", prompt)

    def test_fixed_judge_prompt_preserves_medium_output_and_truncates_large_output(self):
        task = {"id": "task-judge", "role": "task", "prompt": "do", "dimension_targets": ["correctness"]}
        deterministic = {"execution_status": "completed", "outcome": "unknown", "hard_gate_status": "unknown", "checks": []}
        medium_output = "x" * 2000
        medium_prompt = rolebench.build_fixed_judge_prompt(task, {"execution_status": "completed", "output": medium_output}, deterministic)
        self.assertIn(medium_output, medium_prompt)
        large_output = "y" * (rolebench.JUDGE_INPUT_LIMIT + 10)
        large_prompt = rolebench.build_fixed_judge_prompt(task, {"execution_status": "completed", "output": large_output}, deterministic)
        self.assertIn("y" * rolebench.JUDGE_INPUT_LIMIT + "...", large_prompt)
        self.assertNotIn("y" * 10 + '"', large_prompt)

    def test_fixed_judge_cited_paths_strip_line_suffixes_and_dedupe_in_order(self):
        output = (
            "I edited `src/app.ts:12`, then src/app.ts:12:5 and ./src/app.ts again; "
            "scripts/rolebench.py, docs/README.md, lib/util.rs, a/b/c.svelte, and src/app.tsx. "
            "Unrelated tokens: model:high, omp-rolebench/run, 1.5.3, LICENSE, foo.xyz, plain words."
        )
        self.assertEqual(
            rolebench.cited_workspace_paths(output),
            ["src/app.ts", "scripts/rolebench.py", "docs/README.md", "lib/util.rs", "a/b/c.svelte", "src/app.tsx"],
        )
        self.assertEqual(rolebench.cited_workspace_paths("no paths here"), [])

    def test_fixed_judge_prompt_includes_numbered_workspace_excerpts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "src").mkdir()
            (root / "src" / "small.py").write_text("import os\n\nprint(os.getcwd())\n", encoding="utf-8")
            (root / "src" / "app.py").write_text("".join(f"line {number}\n" for number in range(1, 801)), encoding="utf-8")
            task = {
                "id": "task-judge",
                "role": "task",
                "prompt": "do",
                "workspace": str(root),
                "dimension_targets": ["correctness"],
            }
            result = {"execution_status": "completed", "output": "Fixed src/app.py:12 and src/small.py, plus missing/file.py."}
            deterministic = {"execution_status": "completed", "outcome": "pass", "hard_gate_status": "pass", "checks": []}
            prompt = rolebench.build_fixed_judge_prompt(task, result, deterministic)
            instructions, payload_text = prompt.split("\n\n", 1)
            payload = json.loads(payload_text)
            self.assertIn("use it to verify file and line claims", instructions)
            self.assertIn("listed there remain unknown", instructions)
            self.assertNotIn(str(root), prompt)
            self.assertNotIn("workspace", payload["task"])
            self.assertEqual([item["path"] for item in payload["workspace_excerpts"]], ["src/app.py", "src/small.py"])
            small = payload["workspace_excerpts"][1]
            self.assertEqual(small["total_lines"], 3)
            self.assertEqual(small["excerpted_lines"], 3)
            self.assertFalse(small["truncated"])
            self.assertEqual(small["text"], "1: import os\n2: \n3: print(os.getcwd())")
            large = payload["workspace_excerpts"][0]
            self.assertEqual(large["total_lines"], 800)
            self.assertTrue(large["truncated"])
            self.assertEqual(large["excerpted_lines"], len(large["text"].splitlines()))
            self.assertLess(large["excerpted_lines"], 800)
            self.assertLessEqual(len(large["text"]), rolebench.WORKSPACE_EXCERPT_FILE_BYTES)
            self.assertTrue(large["text"].startswith("1: line 1\n2: line 2\n"))
            self.assertTrue(large["text"].endswith(f"{large['excerpted_lines']}: line {large['excerpted_lines']}"))

    def test_fixed_judge_workspace_excerpts_skip_escapes_binaries_and_large_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = root / "workspace"
            (workspace / "src").mkdir(parents=True)
            (workspace / "src" / "app.py").write_text("inside\n", encoding="utf-8")
            (workspace / "node_modules" / "pkg").mkdir(parents=True)
            (workspace / "node_modules" / "pkg" / "index.js").write_text("vendored\n", encoding="utf-8")
            (workspace / "assets").mkdir()
            (workspace / "assets" / "logo.png").write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00binary\n")
            (workspace / "big").mkdir()
            (workspace / "big" / "data.json").write_bytes(b"x" * (rolebench.WORKSPACE_EXCERPT_MAX_FILE_BYTES + 1))
            outside = root / "outside"
            outside.mkdir()
            (outside / "secret.py").write_text("outside\n", encoding="utf-8")
            (workspace / "linked.py").symlink_to(outside / "secret.py")
            output = " ".join([
                "../outside/secret.py",
                str(outside / "secret.py"),
                "linked.py",
                "node_modules/pkg/index.js",
                "assets/logo.png",
                "big/data.json",
                "src/app.py",
            ])
            self.assertEqual(
                [item["path"] for item in rolebench.workspace_excerpts(str(workspace), output)],
                ["src/app.py"],
            )
            self.assertIsNone(
                rolebench.workspace_excerpt_file(workspace.resolve(), str(outside / "secret.py"), rolebench.WORKSPACE_EXCERPT_FILE_BYTES)
            )

    def test_fixed_judge_prompt_without_resolvable_citations_keeps_empty_excerpt_list(self):
        deterministic = {"execution_status": "completed", "outcome": "unknown", "hard_gate_status": "unknown", "checks": []}
        expected_keys = {"task", "role", "execution", "deterministic_evidence", "workspace_excerpts"}
        cases = [
            {"id": "t", "role": "task", "prompt": "do", "dimension_targets": ["correctness"]},
            {"id": "t", "role": "task", "prompt": "do", "workspace": "/nonexistent/rolebench-workspace", "dimension_targets": ["correctness"]},
        ]
        for task in cases:
            prompt = rolebench.build_fixed_judge_prompt(task, {"execution_status": "completed", "output": "src/app.py:3"}, deterministic)
            payload = json.loads(prompt.split("\n\n", 1)[1])
            self.assertEqual(set(payload), expected_keys)
            self.assertEqual(payload["workspace_excerpts"], [])
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "src").mkdir()
            (Path(directory) / "src" / "app.py").write_text("content\n", encoding="utf-8")
            task = {"id": "t", "role": "task", "prompt": "do", "dimension_targets": ["correctness"]}
            uncited = rolebench.build_fixed_judge_prompt(
                {**task, "workspace": directory},
                {"execution_status": "completed", "output": "nothing cited"},
                deterministic,
            )
            without_workspace = rolebench.build_fixed_judge_prompt(task, {"execution_status": "completed", "output": "nothing cited"}, deterministic)
            self.assertEqual(json.loads(uncited.split("\n\n", 1)[1])["workspace_excerpts"], [])
            self.assertEqual(uncited.split("\n\n", 1)[1], without_workspace.split("\n\n", 1)[1])

    def test_fixed_judge_rubric_version_records_workspace_excerpt_contract(self):
        self.assertEqual(rolebench.FIXED_JUDGE_RUBRIC_VERSION, "rolebench-v2-20260924")

    def test_fixed_judge_estimate_reserves_the_full_workspace_excerpt_budget(self):
        pricing = {"judge/model": {"input_per_million": 1.0, "output_per_million": 2.0}}
        assignments = [{"id": "assignment-001", "mode": "all-same", "mapping": {"task": "p/a"}}]
        tasks = [{"id": "task-1", "role": "task", "prompt": "do", "expected": {"output_contains": ["PASS"]}}]
        baseline_prompts = []
        excerpt_prompts = []
        skeleton = rolebench.build_fixed_judge_prompt
        with patch("rolebench.build_fixed_judge_prompt", side_effect=skeleton) as spy:
            estimate = rolebench.estimate_fixed_judge(["task"], assignments, tasks, 1, "judge/model", pricing)
        for call in spy.call_args_list:
            task, result, evaluation, excerpts = call.args
            baseline_prompts.append(skeleton(task, result, evaluation, []))
            excerpt_prompts.append(skeleton(task, result, evaluation, excerpts))
        self.assertEqual(len(excerpt_prompts), 1)
        self.assertEqual(sum(len(item["text"]) for item in spy.call_args_list[0].args[3]), rolebench.WORKSPACE_EXCERPT_TOTAL_BYTES)
        self.assertLessEqual(len(spy.call_args_list[0].args[3]), rolebench.WORKSPACE_EXCERPT_MAX_FILES)
        self.assertEqual(estimate["input_tokens"], math.ceil(len(excerpt_prompts[0]) / 4))
        baseline_tokens = math.ceil(len(baseline_prompts[0]) / 4)
        self.assertGreaterEqual(estimate["input_tokens"] - baseline_tokens, math.ceil(rolebench.WORKSPACE_EXCERPT_TOTAL_BYTES / 4))

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

    def test_delegated_controller_estimate_includes_repeated_calls_and_cost(self):
        pricing = {"p/controller": {"input_per_million": 1.0, "output_per_million": 2.0}}
        assignments = [{"id": "assignment-001", "mode": "all-same", "mapping": {"task": "p/a"}}]
        tasks = [{"id": "task-1", "role": "task", "prompt": "do", "max_output_tokens": 400}]
        estimate = rolebench.estimate_controller(assignments, tasks, 3, "p/controller:high", pricing)
        self.assertTrue(estimate["enabled"])
        self.assertEqual(estimate["calls"], 3)
        self.assertEqual(estimate["model"], "p/controller:high")
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

    def test_fixed_judge_normalizes_observed_equivalent_schemas(self):
        requested = ["correctness", "honesty"]
        item = {"status": "pass", "score": 2, "evidence": ["supported"]}
        variants = [
            {"dimensions": {"correctness": item, "honesty": item}},
            {"dimensions": [{"name": "correctness", **item}, {"name": "honesty", **item}]},
            {"dimensions": [{"dimension": "correctness", **item}, {"dimension": "honesty", **item}]},
            {"correctness": item, "honesty": item},
            {"evaluation": {"correctness": item, "honesty": item}, "overall": {"summary": "ok"}},
        ]
        for variant in variants:
            with self.subTest(keys=list(variant)):
                normalized = rolebench.normalize_fixed_judge_dimensions(variant, requested)
                self.assertEqual(set(normalized), set(requested))
                self.assertEqual(normalized["correctness"]["status"], "pass")
        fenced = "```json\n" + json.dumps(variants[1]) + "\n```"
        self.assertEqual(rolebench.parse_fixed_judge_document(fenced), variants[1])
        with self.assertRaisesRegex(rolebench.RoleBenchError, "duplicate dimension"):
            rolebench.normalize_fixed_judge_dimensions(
                {"dimensions": [{"name": "correctness", **item}, {"dimension": "correctness", **item}]},
                requested,
            )

    @patch("rolebench.subprocess.run")
    def test_fixed_judge_keeps_raw_result_separate_from_evaluation(self, run):
        judge_payload = {
            "dimensions": {"correctness": {"status": "pass", "score": 2, "evidence": ["output matched"]}},
            "summary": "matched the declared contract",
        }
        captured = {}

        def complete(command, **_kwargs):
            config = Path(command[command.index("--config") + 1])
            cwd = Path(command[command.index("--cwd") + 1])
            captured["config"] = config.read_text(encoding="utf-8")
            captured["same_root"] = config.parent == cwd
            return SimpleNamespace(returncode=0, stdout=json.dumps({"text": json.dumps(judge_payload)}), stderr="")

        run.side_effect = complete
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
        thinking_index = command.index("--thinking")
        config_index = command.index("--config")
        cwd_index = command.index("--cwd")
        self.assertEqual(command[thinking_index + 1], "high")
        self.assertEqual(config_index, thinking_index + 2)
        self.assertEqual(cwd_index, config_index + 2)
        self.assertTrue(command[cwd_index + 1])
        self.assertEqual(command[cwd_index + 2], "--no-tools")
        self.assertIn("--no-extensions", command)
        self.assertIn("--no-skills", command)
        self.assertIn("--no-rules", command)
        self.assertTrue(captured["same_root"])
        self.assertIn('default: "judge/model"', captured["config"])

    @patch("rolebench.subprocess.run")
    def test_fixed_judge_rebuild_uses_the_exact_accepted_payload(self, run):
        judge_payload = {
            "text": "human-readable summary",
            "dimensions": {
                "correctness": {"status": "pass", "score": 2, "evidence": ["matched"]},
            },
            "summary": "accepted",
        }
        run.return_value = SimpleNamespace(returncode=0, stdout=json.dumps(judge_payload), stderr="")
        task = {"id": "task-judge", "role": "task", "prompt": "do", "dimension_targets": ["correctness"]}
        result = {"assignment_id": "a", "task_id": "task-judge", "role": "task", "model": "p/a", "replicate": 1, "execution_status": "completed", "output": "PASS"}
        deterministic = {"execution_status": "completed", "outcome": "pass", "hard_gate_status": "pass", "checks": []}
        raw, evaluation = rolebench.run_fixed_judge("omp", "judge/model", task, result, deterministic, timeout=5)
        rebuilt = rolebench.rebuild_judge_evaluation(task, result, raw)
        self.assertEqual(raw["output"], "human-readable summary")
        self.assertEqual(raw["evaluation_payload"], judge_payload)
        self.assertEqual(
            rolebench.judge_evaluation_audit_payload(rebuilt),
            rolebench.judge_evaluation_audit_payload(evaluation),
        )

    @patch("rolebench.subprocess.run")
    def test_fixed_judge_reads_final_assistant_event_and_list_dimensions(self, run):
        judge_payload = {
            "dimensions": [
                {"dimension": "correctness", "status": "pass", "score": 2, "evidence": ["matched"]},
            ]
        }
        events = [
            {"type": "message_end", "message": {"role": "user", "content": [{"type": "text", "text": "PROMPT"}], "usage": {"cost": {"total": 0}}}},
            {"type": "message_start", "message": {"role": "assistant", "content": [{"type": "thinking", "thinking": "private"}]}},
            {"type": "message_update", "assistantMessageEvent": {"type": "text_start", "contentIndex": 1}},
            {"type": "message_update", "assistantMessageEvent": {"type": "text_delta", "contentIndex": 1, "delta": "```json\n"}},
            {"type": "message_update", "assistantMessageEvent": {"type": "text_delta", "contentIndex": 1, "delta": json.dumps(judge_payload)}},
            {"type": "message_update", "assistantMessageEvent": {"type": "text_end", "contentIndex": 1, "content": "```json\n" + json.dumps(judge_payload) + "\n```"}},
            {"type": "message_end", "message": {"role": "assistant", "content": [{"type": "thinking", "thinking": "private"}], "usage": {"input": 5, "output": 2, "totalTokens": 7, "cost": {"total": 0.01}}}},
        ]
        run.return_value = SimpleNamespace(returncode=0, stdout="\n".join(json.dumps(event) for event in events), stderr="")
        task = {"id": "task-judge", "role": "task", "prompt": "do", "dimension_targets": ["correctness"]}
        result = {"assignment_id": "a", "task_id": "task-judge", "role": "task", "model": "p/a", "replicate": 1, "execution_status": "completed", "output": "PASS"}
        deterministic = {"execution_status": "completed", "outcome": "pass", "hard_gate_status": "pass", "checks": []}
        raw, evaluation = rolebench.run_fixed_judge("omp", "judge/model", task, result, deterministic, timeout=5)
        self.assertEqual(raw["status"], "pass")
        self.assertNotIn("PROMPT", raw["output"])
        self.assertEqual(rolebench.extract_cost(raw["usage"]), 0.01)
        self.assertEqual(evaluation["dimensions"]["correctness"]["status"], "pass")

    def test_omp_event_stream_prefers_last_finalized_text_segment(self):
        final = '{"dimensions":{"correctness":{"status":"pass","score":2,"evidence":["matched"]}}}'
        payload = [
            {"type": "message_start", "message": {"role": "assistant", "content": []}},
            {"type": "message_update", "assistantMessageEvent": {"type": "text_end", "contentIndex": 0, "content": "I will inspect the workspace."}},
            {"type": "message_update", "assistantMessageEvent": {"type": "text_end", "contentIndex": 2, "content": final}},
            {
                "type": "message_end",
                "message": {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "I will inspect the workspace."},
                        {"type": "toolCall", "name": "read"},
                        {"type": "text", "text": final},
                    ],
                },
            },
        ]
        self.assertEqual(rolebench.extract_output(payload), final)

    def test_omp_event_stream_joins_deltas_when_text_end_omits_content(self):
        payload = [
            {"type": "message_start", "message": {"role": "assistant", "content": []}},
            {"type": "message_update", "assistantMessageEvent": {"type": "text_start", "contentIndex": 0}},
            {"type": "message_update", "assistantMessageEvent": {"type": "text_delta", "contentIndex": 0, "delta": "hel"}},
            {"type": "message_update", "assistantMessageEvent": {"type": "text_delta", "contentIndex": 0, "delta": "lo"}},
            {"type": "message_update", "assistantMessageEvent": {"type": "text_end", "contentIndex": 0}},
            {"type": "message_end", "message": {"role": "assistant", "content": []}},
        ]
        self.assertEqual(rolebench.extract_output(payload), "hello")

    def test_omp_event_stream_discards_aborted_assistant_turn_segments(self):
        payload = [
            {"type": "message_start", "message": {"role": "assistant", "content": []}},
            {"type": "message_update", "assistantMessageEvent": {"type": "text_end", "contentIndex": 0, "content": "OLD"}},
            {"type": "message_start", "message": {"role": "assistant", "content": []}},
            {"type": "message_end", "message": {"role": "assistant", "content": [{"type": "text", "text": "NEW"}]}},
        ]
        self.assertEqual(rolebench.extract_output(payload), "NEW")

    @patch("rolebench.subprocess.run", side_effect=OSError("omp is missing"))
    def test_fixed_judge_marks_call_started_only_before_subprocess(self, run):
        task = {"id": "task-judge", "role": "task", "prompt": "do", "dimension_targets": ["correctness"]}
        result = {"assignment_id": "a", "task_id": "task-judge", "role": "task", "model": "p/a", "replicate": 1, "execution_status": "completed", "output": "PASS"}
        deterministic = {"execution_status": "completed", "outcome": "pass", "hard_gate_status": "pass", "checks": []}
        raw, evaluation = rolebench.run_fixed_judge("omp", "judge/model", task, result, deterministic, timeout=5)
        self.assertFalse(raw["provider_call_started"])
        self.assertEqual(raw["error_type"], "judge_invalid")
        self.assertEqual(evaluation["outcome"], "unknown")
        run.assert_called_once()

    def test_recommendations_require_repeated_hard_gate_evidence(self):
        task = {"id": "task-1", "role": "task", "dimension_targets": ["correctness", "instruction_following"]}
        results = []
        evaluations = []
        judge_results = []
        judge_evaluations = []
        for replicate in range(1, 4):
            identity = {"assignment_id": "assignment-001", "task_id": "task-1", "replicate": replicate}
            results.append({**identity, "role": "task", "model": "p/a", "status": "pass", "latency_ms": replicate, "usage": {"cost": {"total": 0.001}}})
            evaluations.append({**identity, "role": "task", "model": "p/a", "execution_status": "completed", "outcome": "pass", "hard_gate_status": "pass", "dimensions": {"correctness": {"status": "pass", "score": 2, "hard_gate": True}}})
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
        coverage = rolebench.build_recommendations(
            ["task"],
            [task, {"id": "task-2", "role": "task", "dimension_targets": ["correctness"]}],
            results,
            evaluations,
            judge_results,
            judge_evaluations,
            3,
        )
        self.assertEqual(coverage["status"], "blocked")
        self.assertIn("task-2", coverage["candidates"][0]["blocked_reasons"][0])

    def test_recommendations_allow_judge_to_cover_hard_dimension_without_deterministic_evaluator(self):
        identity = {"assignment_id": "assignment-001", "task_id": "task-1", "replicate": 1}
        task = {"id": "task-1", "role": "task", "dimension_targets": ["correctness", "honesty"]}
        results = [{**identity, "role": "task", "model": "p/a", "status": "pass"}]
        evaluations = [{
            **identity,
            "role": "task",
            "model": "p/a",
            "execution_status": "completed",
            "outcome": "unknown",
            "hard_gate_status": "unknown",
            "checks": [],
            "dimensions": {
                "correctness": {"status": "pass", "score": 2, "hard_gate": True, "evaluator": "deterministic"},
                "honesty": {"status": "unknown", "score": None, "hard_gate": True, "evaluator": "none"},
            },
        }]
        judge_results = [{**identity, "model": "judge/model"}]
        judge_evaluations = [{
            **identity,
            "role": "task",
            "model": "p/a",
            "hard_gate_status": "pass",
            "dimensions": {
                "correctness": {"status": "pass", "score": 2, "hard_gate": True},
                "honesty": {"status": "pass", "score": 2, "hard_gate": True},
            },
        }]
        recommendation = rolebench.build_recommendations(
            ["task"], [task], results, evaluations, judge_results, judge_evaluations, 1
        )
        self.assertEqual(recommendation["status"], "complete")
        self.assertEqual(recommendation["candidates"][0]["deterministic_hard_gate"], "pass")

        evaluations[0]["dimensions"]["honesty"]["evaluator"] = "deterministic"
        blocked = rolebench.build_recommendations(
            ["task"], [task], results, evaluations, judge_results, judge_evaluations, 1
        )
        self.assertEqual(blocked["status"], "blocked")
        self.assertTrue(any("not fully verified" in reason for reason in blocked["candidates"][0]["blocked_reasons"]))

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

    def test_task_schema_accepts_scenario_and_task_type(self):
        normalized = rolebench.normalize_task({
            "id": "task-1", "role": "task", "scenario": "repair a local script",
            "task_type": "code-repair", "prompt": "fix it",
        })
        self.assertEqual(normalized["scenario"], "repair a local script")
        self.assertEqual(normalized["task_type"], "code-repair")
        with self.assertRaises(rolebench.RoleBenchError):
            rolebench.normalize_task({"id": "task-1", "role": "task", "scenario": "", "prompt": "fix it"})

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

    @patch("rolebench.subprocess.run", side_effect=OSError("omp is missing"))
    def test_run_one_marks_provider_call_started_only_after_spawn(self, run):
        assignment = {"id": "a", "mode": "role-isolated", "role": "task", "candidate": "p/a", "mapping": {"task": "p/a"}}
        task = {"id": "task-001", "role": "task", "prompt": "do it"}
        row = rolebench.run_one("omp", assignment, task, 1, Path("."), 5, Path("."), "direct", None)
        self.assertFalse(row["provider_call_started"])
        self.assertEqual(row["error_type"], "executor_setup")
        run.assert_called_once()

    @patch("rolebench.subprocess.run")
    def test_delegated_run_uses_project_probe_agent(self, run):
        run.return_value = SimpleNamespace(returncode=0, stdout=json.dumps({"text": "<task-result status=\"completed\">PASS</task-result>"}), stderr="")
        assignment = {"id": "a", "mode": "role-isolated", "role": "task", "candidate": "p/a", "mapping": {"task": "p/a"}, "thinking": {"task": "high"}}
        task = {"id": "task-001", "role": "task", "prompt": "do it"}
        row = rolebench.run_one("omp", assignment, task, 1, Path("."), 5, Path("."), "delegated", "p/controller:high")
        self.assertEqual(row["status"], "pass")
        self.assertEqual(row["controller_model"], "p/controller")
        self.assertEqual(row["controller_thinking"], "high")
        command = run.call_args.args[0]
        self.assertIn("--tools", command)
        self.assertIn("task", command)
        self.assertEqual(command[command.index("--model") + 1], "p/controller")
        self.assertEqual(command[command.index("--thinking") + 1], "high")
        prompt = command[-1]
        self.assertIn("do not background it", prompt)
        self.assertIn("keep waiting until the result is delivered", prompt)

    def test_tasks_file_resolves_relative_workspace_from_suite_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            suite = root / "suite" / "tasks.jsonl"
            workspace = suite.parent / "fixtures" / "work"
            workspace.mkdir(parents=True)
            suite.write_text(
                json.dumps({
                    "schema_version": 1,
                    "id": "portable-task",
                    "role": "task",
                    "prompt": "do it",
                    "workspace": "fixtures/work",
                }) + "\n",
                encoding="utf-8",
            )
            args = SimpleNamespace(tasks_file=str(suite), task_source="core", suite="none")
            tasks = rolebench.load_tasks(args, ["task"])
            self.assertEqual(tasks[0]["workspace"], str(workspace.resolve()))

    @patch("rolebench.subprocess.run")
    def test_delegated_run_requires_resolved_controller(self, run):
        run.return_value = SimpleNamespace(returncode=0, stdout=json.dumps({"text": "PASS"}), stderr="")
        assignment = {"id": "a", "mode": "role-isolated", "role": "task", "candidate": "p/a", "mapping": {"task": "p/a"}, "thinking": {"task": "high"}}
        task = {"id": "task-001", "role": "task", "prompt": "do it"}
        row = rolebench.run_one("omp", assignment, task, 1, Path("."), 5, Path("."), "delegated", None)
        self.assertEqual(row["error_type"], "executor_setup")
        self.assertIn("resolved controller model", row["error"])
        run.assert_not_called()

    def test_delegated_controller_defaults_to_configured_default_role(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "config.yml"
            args = self.make_run_args(root, config_file=str(config), executor="delegated", execute=False, confirm=False)
            config.write_text("modelRoles:\n  default: p/controller:high\n  task: p/base\n", encoding="utf-8")
            with patch.object(rolebench, "load_tasks", return_value=[{"id": "task-1", "role": "task", "prompt": "do", "dimension_targets": ["correctness"]}]):
                self.assertEqual(rolebench.run_command(args), 0)
            manifest = json.loads((root / "run" / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["controller_model"], "p/controller:high")
            self.assertEqual(manifest["controller_model_source"], "default-role")
            self.assertIn("controller: p/controller:high (default role;", rolebench.format_plan(manifest))

    def test_delegated_controller_must_be_explicit_when_default_role_is_missing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = self.make_run_args(root, executor="delegated", execute=False, confirm=False)
            with patch.object(rolebench, "load_tasks", return_value=[{"id": "task-1", "role": "task", "prompt": "do", "dimension_targets": ["correctness"]}]):
                with self.assertRaisesRegex(rolebench.RoleBenchError, "modelRoles.default"):
                    rolebench.run_command(args)

    def test_controller_model_is_rejected_for_direct_executor(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = self.make_run_args(root, controller_model="p/controller", execute=False, confirm=False)
            with patch.object(rolebench, "load_tasks", return_value=[{"id": "task-1", "role": "task", "prompt": "do", "dimension_targets": ["correctness"]}]):
                with self.assertRaisesRegex(rolebench.RoleBenchError, "requires --executor delegated"):
                    rolebench.run_command(args)

    def test_delegated_controller_cost_is_included_in_budget_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "config.yml"
            pricing = root / "pricing.json"
            pricing.write_text(json.dumps({
                "p/a": {"input_per_million": 1, "output_per_million": 1},
                "p/controller": {"input_per_million": 1, "output_per_million": 1},
            }), encoding="utf-8")
            args = self.make_run_args(
                root,
                config_file=str(config),
                executor="delegated",
                pricing_file=str(pricing),
                max_cost_usd=0,
            )
            config.write_text("modelRoles:\n  default: p/controller\n  task: p/base\n", encoding="utf-8")
            task = {"id": "task-1", "role": "task", "prompt": "do", "dimension_targets": ["correctness"]}
            with patch.object(rolebench, "load_tasks", return_value=[task]):
                with patch.object(rolebench, "run_one") as run:
                    self.assertEqual(rolebench.run_command(args), 2)
            run.assert_not_called()
            summary = json.loads((root / "run" / "summary.json").read_text(encoding="utf-8"))
            self.assertGreater(summary["controller_cost_usd"], 0)
            self.assertEqual(
                summary["estimated_cost_usd"],
                summary["model_cost_usd"] + summary["controller_cost_usd"] + summary["judge_cost_usd"],
            )

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

    def test_retried_cell_preserves_orphaned_artifact_attempt(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            artifacts = output / "artifacts"
            snapshot = {"complete": True, "files": {}, "errors": [], "incomplete": []}
            workspace_diff = {"created": [], "modified": [], "deleted": [], "changed": [], "complete": True, "errors": [], "text_unavailable": [], "patch": "", "patch_truncated": False}
            first = rolebench.write_workspace_artifacts(output, artifacts, "same-cell", snapshot, snapshot, workspace_diff)
            second = rolebench.write_workspace_artifacts(output, artifacts, "same-cell", snapshot, snapshot, workspace_diff)
            self.assertNotEqual(first["workspace_before"], second["workspace_before"])
            self.assertIn("attempt-2", second["workspace_before"])
            self.assertTrue((output / first["workspace_before"]).is_file())
            self.assertTrue((output / second["workspace_before"]).is_file())

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
                    self.assertEqual(rolebench.run_command(args), 130)
            lines = (root / "run" / "results.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(lines), 1)
            self.assertEqual(json.loads(lines[0])["output"], "first")
            summary = json.loads((root / "run" / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["status"], "aborted")
            self.assertEqual(summary["stopped_reason"], "interrupted")

    def test_resume_skips_persisted_cell_and_completes_missing_cell(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            task = {"id": "task-1", "role": "task", "prompt": "do", "dimension_targets": ["correctness"]}
            first_evaluation = {"schema_version": 1, "assignment_id": "assignment-001-01", "task_id": "task-1", "replicate": 1, "role": "task", "model": "p/a", "thinking": None, "execution_status": "completed", "outcome": "pass", "hard_gate_status": "pass", "checks": [], "dimensions": {}}
            first = {"status": "pass", "execution_status": "completed", "provider_call_started": True, "output": "first", "_evaluation": first_evaluation}
            with patch.object(rolebench, "load_tasks", return_value=[task]):
                with patch.object(rolebench, "run_one", side_effect=[first, KeyboardInterrupt]):
                    self.assertEqual(rolebench.run_command(self.make_run_args(root, runs=2)), 130)
            second_evaluation = {"schema_version": 1, "assignment_id": "assignment-001-01", "task_id": "task-1", "replicate": 2, "role": "task", "model": "p/a", "thinking": None, "execution_status": "completed", "outcome": "pass", "hard_gate_status": "pass", "checks": [], "dimensions": {}}
            second = {"status": "pass", "execution_status": "completed", "provider_call_started": True, "output": "second", "_evaluation": second_evaluation}
            with patch.object(rolebench, "load_tasks", return_value=[task]):
                with patch.object(rolebench, "run_one", return_value=second) as run:
                    self.assertEqual(rolebench.run_command(self.make_run_args(root, runs=2, resume=True)), 0)
            rows = rolebench.load_jsonl(root / "run" / "results.jsonl", "results.jsonl")
            summary = json.loads((root / "run" / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(run.call_count, 1)
            self.assertEqual([row["output"] for row in rows], ["first", "second"])
            self.assertEqual(summary["status"], "completed")
            self.assertTrue(summary["resumed"])
            self.assertEqual(summary["skipped_cells"], 1)
            self.assertFalse(list((root / "run").glob("results.jsonl.previous-*")))

    def test_resume_after_judge_interruption_does_not_repeat_candidate_call(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pricing = root / "pricing.json"
            pricing.write_text(json.dumps({
                "p/a": {"input_per_million": 1, "output_per_million": 1},
                "judge/model": {"input_per_million": 1, "output_per_million": 1},
            }), encoding="utf-8")
            task = {"id": "task-1", "role": "task", "prompt": "do", "dimension_targets": ["correctness"]}
            evaluation = {
                "schema_version": 1,
                "assignment_id": "assignment-001-01",
                "task_id": "task-1",
                "replicate": 1,
                "role": "task",
                "model": "p/a",
                "thinking": None,
                "execution_status": "completed",
                "outcome": "pass",
                "hard_gate_status": "pass",
                "checks": [],
                "dimensions": {},
            }
            candidate = {
                "status": "pass",
                "execution_status": "completed",
                "provider_call_started": True,
                "output": "candidate answer",
                "_evaluation": evaluation,
            }
            initial_args = self.make_run_args(root, pricing_file=str(pricing), judge_model="judge/model")
            with patch.object(rolebench, "load_tasks", return_value=[task]):
                with patch.object(rolebench, "run_one", return_value=candidate) as candidate_call:
                    with patch.object(rolebench, "run_fixed_judge", side_effect=KeyboardInterrupt):
                        self.assertEqual(rolebench.run_command(initial_args), 130)
            self.assertEqual(candidate_call.call_count, 1)
            self.assertEqual(len(rolebench.load_jsonl(root / "run" / "results.jsonl", "results.jsonl")), 1)
            self.assertFalse((root / "run" / "judge_results.jsonl").exists())

            judge_result = {
                "judge_result_id": "assignment-001-01:task-1:1",
                "assignment_id": "assignment-001-01",
                "task_id": "task-1",
                "role": "task",
                "model": "judge/model",
                "replicate": 1,
                "provider_call_started": True,
                "status": "pass",
                "execution_status": "completed",
                "output": '{"dimensions":{"correctness":{"status":"pass","score":2,"evidence":"ok"}}}',
            }
            judge_evaluation = {
                "assignment_id": "assignment-001-01",
                "task_id": "task-1",
                "replicate": 1,
                "role": "task",
                "model": "p/a",
                "execution_status": "completed",
                "outcome": "pass",
                "hard_gate_status": "pass",
                "dimensions": {"correctness": {"status": "pass", "score": 2, "evidence": "ok", "evaluator": "fixed_judge"}},
            }
            resumed_args = self.make_run_args(root, pricing_file=str(pricing), judge_model="judge/model", resume=True)
            with patch.object(rolebench, "load_tasks", return_value=[task]):
                with patch.object(rolebench, "run_one") as candidate_call:
                    with patch.object(rolebench, "run_fixed_judge", return_value=(judge_result, judge_evaluation)) as judge_call:
                        self.assertEqual(rolebench.run_command(resumed_args), 0)
            candidate_call.assert_not_called()
            self.assertEqual(judge_call.call_count, 1)
            summary = json.loads((root / "run" / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["status"], "completed")
            self.assertEqual(summary["skipped_cells"], 1)
            self.assertEqual(summary["result_rows"], 1)
            self.assertEqual(summary["judge_result_rows"], 1)
            self.assertEqual(summary["judge_evaluation_rows"], 1)

    def test_resume_rejects_changed_experiment_or_harness(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            task = {"id": "task-1", "role": "task", "prompt": "do", "dimension_targets": ["correctness"]}
            row = {"status": "pass", "execution_status": "completed", "provider_call_started": False, "output": "done"}
            with patch.object(rolebench, "load_tasks", return_value=[task]):
                with patch.object(rolebench, "run_one", return_value=row):
                    self.assertEqual(rolebench.run_command(self.make_run_args(root)), 0)
            with patch.object(rolebench, "load_tasks", return_value=[task]):
                with self.assertRaisesRegex(rolebench.RoleBenchError, "experiment does not match"):
                    rolebench.run_command(self.make_run_args(root, runs=2, resume=True))
            with patch.object(rolebench, "load_tasks", return_value=[task]):
                with patch.object(rolebench, "harness_fingerprint", return_value={"algorithm": "sha256", "digest": "changed", "file_count": 1}):
                    with self.assertRaisesRegex(rolebench.RoleBenchError, "harness differs"):
                        rolebench.run_command(self.make_run_args(root, resume=True))

    def test_runtime_harness_change_stops_after_current_cell(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            task = {"id": "task-1", "role": "task", "prompt": "do", "dimension_targets": ["correctness"]}
            row = {"status": "pass", "execution_status": "completed", "provider_call_started": False, "output": "done"}
            baseline = {"algorithm": "sha256", "digest": "same", "file_count": 1}
            changed = {"algorithm": "sha256", "digest": "changed", "file_count": 1}
            with patch.object(rolebench, "load_tasks", return_value=[task]):
                with patch.object(rolebench, "harness_fingerprint", side_effect=[baseline, baseline, changed]):
                    with patch.object(rolebench, "run_one", return_value=row):
                        self.assertEqual(rolebench.run_command(self.make_run_args(root, runs=2)), 4)
            summary = json.loads((root / "run" / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["status"], "aborted")
            self.assertEqual(summary["stopped_reason"], "harness_integrity_changed")
            self.assertEqual(summary["harness_status"], "changed")
            self.assertEqual(summary["result_rows"], 1)

    def test_cell_internal_error_is_recorded_and_stops_the_run(self):
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
                    self.assertEqual(rolebench.run_command(args), 4)
            rows = [json.loads(line) for line in (root / "run" / "results.jsonl").read_text(encoding="utf-8").splitlines()]
            summary = json.loads((root / "run" / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(run.call_count, 1)
            self.assertEqual(rows[0]["error_type"], "harness_internal")
            self.assertEqual(summary["status"], "aborted")
            self.assertEqual(summary["stopped_reason"], "harness_internal")
            self.assertEqual(summary["harness_status"], "failed")
            self.assertEqual(summary["result_rows"], 1)
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

    def test_model_error_continues_uncapped_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = self.make_run_args(root, runs=2)
            error_row = {"status": "error", "execution_status": "failed", "error_type": "omp_exit", "provider_call_started": True}
            pass_row = {"status": "pass", "execution_status": "completed", "provider_call_started": True, "output": "done"}
            with patch.object(rolebench, "load_tasks", return_value=[{"id": "task-1", "role": "task", "prompt": "do"}]):
                with patch.object(rolebench, "run_one", side_effect=[error_row, pass_row]) as run:
                    self.assertEqual(rolebench.run_command(args), 0)
            summary = json.loads((root / "run" / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(run.call_count, 2)
            self.assertEqual(summary["status"], "completed")
            self.assertEqual(summary["result_rows"], 2)

    def test_executor_setup_error_stops_as_harness_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            error_row = {"status": "error", "execution_status": "failed", "error_type": "executor_setup", "provider_call_started": False}
            task = {"id": "task-1", "role": "task", "prompt": "do"}
            with patch.object(rolebench, "load_tasks", return_value=[task]):
                with patch.object(rolebench, "run_one", return_value=error_row) as run:
                    self.assertEqual(rolebench.run_command(self.make_run_args(root, runs=2)), 4)
            summary = json.loads((root / "run" / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(run.call_count, 1)
            self.assertEqual(summary["status"], "aborted")
            self.assertEqual(summary["stopped_reason"], "harness_execution_failed")
            (root / "run" / "summary.json").unlink()
            with patch.object(rolebench, "load_tasks", return_value=[task]):
                with patch.object(rolebench, "run_one") as resumed_call:
                    with self.assertRaisesRegex(rolebench.RoleBenchError, "cannot resume a harness-failed run"):
                        rolebench.run_command(self.make_run_args(root, runs=2, resume=True))
            resumed_call.assert_not_called()

    def test_unexpected_fixed_judge_error_is_recorded_and_stops(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pricing = root / "pricing.json"
            pricing.write_text(json.dumps({
                "p/a": {"input_per_million": 1, "output_per_million": 1},
                "judge/model": {"input_per_million": 1, "output_per_million": 1},
            }), encoding="utf-8")
            task = {"id": "task-1", "role": "task", "prompt": "do", "dimension_targets": ["correctness"]}
            evaluation = {"assignment_id": "assignment-001-01", "task_id": "task-1", "replicate": 1, "role": "task", "model": "p/a", "execution_status": "completed", "outcome": "pass", "hard_gate_status": "pass", "checks": [], "dimensions": {}}
            row = {"status": "pass", "execution_status": "completed", "provider_call_started": True, "output": "done", "usage": {"cost": {"total": 0.001}}, "_evaluation": evaluation}
            args = self.make_run_args(root, pricing_file=str(pricing), judge_model="judge/model")
            with patch.object(rolebench, "load_tasks", return_value=[task]):
                with patch.object(rolebench, "run_one", return_value=row):
                    with patch.object(rolebench, "run_fixed_judge", side_effect=RuntimeError("judge harness broke")):
                        self.assertEqual(rolebench.run_command(args), 4)
            summary = json.loads((root / "run" / "summary.json").read_text(encoding="utf-8"))
            judge_row = rolebench.load_jsonl(root / "run" / "judge_results.jsonl", "judge_results.jsonl")[0]
            self.assertEqual(summary["status"], "aborted")
            self.assertEqual(summary["stopped_reason"], "harness_judge_failed")
            self.assertEqual(judge_row["error_type"], "judge_internal")

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
            self.assertAlmostEqual(summary["estimated_cost_usd"], summary["model_cost_usd"] + summary["judge_cost_usd"], places=7)

    def test_fixed_judge_unknown_price_blocks_execution_without_cap(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = self.make_run_args(root, judge_model="judge/model")
            task = {"id": "task-1", "role": "task", "prompt": "do", "dimension_targets": ["correctness"]}
            with patch.object(rolebench, "load_tasks", return_value=[task]):
                with patch.object(rolebench, "run_one") as run:
                    self.assertEqual(rolebench.run_command(args), 2)
            run.assert_not_called()
            summary = json.loads((root / "run" / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["reason"], "judge_cost_unknown")

    def test_recommendation_defaults_to_configured_default_role_judge(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "config.yml"
            args = self.make_run_args(root, config_file=str(config), judge_model=None, recommend=True, execute=False, confirm=False)
            config.write_text("modelRoles:\n  default: judge/default:high\n  task: p/base\n", encoding="utf-8")
            with patch.object(rolebench, "load_tasks", return_value=[{"id": "task-1", "role": "task", "prompt": "do", "dimension_targets": ["correctness"]}]):
                self.assertEqual(rolebench.run_command(args), 0)
            manifest = json.loads((root / "run" / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["judge_model"], "judge/default:high")
            self.assertEqual(manifest["judge_model_source"], "default-role")
            self.assertIn("judge/default:high (default role)", rolebench.format_plan(manifest))

    def test_explicit_judge_model_overrides_default_role(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "config.yml"
            args = self.make_run_args(root, config_file=str(config), judge_model="judge/override", recommend=True, execute=False, confirm=False)
            config.write_text("modelRoles:\n  default: judge/default\n  task: p/base\n", encoding="utf-8")
            with patch.object(rolebench, "load_tasks", return_value=[{"id": "task-1", "role": "task", "prompt": "do", "dimension_targets": ["correctness"]}]):
                self.assertEqual(rolebench.run_command(args), 0)
            manifest = json.loads((root / "run" / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["judge_model"], "judge/override")
            self.assertEqual(manifest["judge_model_source"], "explicit")

    def test_recommendation_requires_default_role_when_judge_is_not_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "config.yml"
            args = self.make_run_args(root, config_file=str(config), judge_model=None, recommend=True, execute=False, confirm=False)
            config.write_text("modelRoles:\n  task: p/base\n", encoding="utf-8")
            with patch.object(rolebench, "load_tasks", return_value=[{"id": "task-1", "role": "task", "prompt": "do", "dimension_targets": ["correctness"]}]):
                with self.assertRaisesRegex(rolebench.RoleBenchError, "modelRoles.default"):
                    rolebench.run_command(args)

    def test_recommendation_preview_does_not_write_unexecuted_result(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = self.make_run_args(root, execute=False, confirm=False, judge_model="judge/model", recommend=True)
            with patch.object(rolebench, "load_tasks", return_value=[{"id": "task-1", "role": "task", "prompt": "do", "dimension_targets": ["correctness"]}]):
                self.assertEqual(rolebench.run_command(args), 0)
            self.assertFalse((root / "run" / "recommendations.json").exists())
            summary = json.loads((root / "run" / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["recommendations"], "not-executed")
            self.assertEqual(summary["reports"]["status"], "generated")
            report = (root / "run" / "report.md").read_text(encoding="utf-8")
            self.assertIn("## Benchmark Scope", report)
            self.assertIn("task-1", report)

    def test_cli_rejects_invalid_run_limits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "config.yml"
            config.write_text("modelRoles:\n  task: p/base\n", encoding="utf-8")
            common = ["run", "--models", "p/a", "--roles", "task", "--config-file", str(config), "--output", str(root / "run")]
            for option, value in (("--runs", "0"), ("--jobs", "0"), ("--timeout", "0"), ("--max-cost-usd", "-1")):
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
            old_report_markdown = output / "report.md"
            old_report_json = output / "report.json"
            old_judge.write_text('{"model":"judge/old"}\n', encoding="utf-8")
            old_evaluation.write_text('{"evaluator":"fixed_judge"}\n', encoding="utf-8")
            old_recommendation.write_text('{"status":"complete"}\n', encoding="utf-8")
            old_report_markdown.write_text("old report\n", encoding="utf-8")
            old_report_json.write_text('{"old":true}\n', encoding="utf-8")
            self.assertEqual(rolebench.main(["run", "--models", "p/new", "--roles", "task", "--config-file", str(config), "--output", str(output)]), 0)
            self.assertFalse(old_judge.exists())
            self.assertFalse(old_evaluation.exists())
            self.assertFalse(old_recommendation.exists())
            self.assertTrue(old_report_markdown.exists())
            self.assertTrue(old_report_json.exists())
            self.assertIn("# OMP RoleBench Report", old_report_markdown.read_text(encoding="utf-8"))
            self.assertIn('"manifest"', old_report_json.read_text(encoding="utf-8"))
            self.assertEqual(len(list(output.glob("judge_results.jsonl.previous-*"))), 1)
            self.assertEqual(len(list(output.glob("judge_evaluations.jsonl.previous-*"))), 1)
            self.assertEqual(len(list(output.glob("recommendations.json.previous-*"))), 1)
            self.assertEqual(len(list(output.glob("report.md.previous-*"))), 1)
            self.assertEqual(len(list(output.glob("report.json.previous-*"))), 1)

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

    def test_report_dimensions_table_splits_deterministic_and_judge_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "manifest.json").write_text(json.dumps({"generated_at": "now", "mode": "role-isolated", "roles": ["task"], "assignments": []}), encoding="utf-8")
            (root / "results.jsonl").write_text("\n".join(json.dumps({
                "assignment_id": "a", "task_id": "task-1", "replicate": replicate, "role": "task", "model": "p/a", "thinking": "high",
                "status": "pass", "latency_ms": 5, "output": "done",
            }) for replicate in (1, 2)) + "\n", encoding="utf-8")
            (root / "evaluations.jsonl").write_text("\n".join(json.dumps({
                "assignment_id": "a", "task_id": "task-1", "replicate": replicate, "role": "task", "model": "p/a", "thinking": "high",
                "outcome": "pass",
                "dimensions": {
                    "correctness": {"status": status, "score": score, "evaluator": "deterministic"},
                    "honesty": {"status": "unknown", "score": None, "evaluator": "none"},
                },
            }) for replicate, status, score in ((1, "pass", 2), (2, "fail", 0))) + "\n", encoding="utf-8")
            (root / "judge_evaluations.jsonl").write_text("\n".join(json.dumps({
                "assignment_id": "a", "task_id": "task-1", "replicate": replicate, "role": "task", "model": "p/a", "thinking": "high",
                "outcome": "pass",
                "dimensions": {
                    "correctness": {"status": status, "score": score, "hard_gate": True, "evaluator": "fixed_judge"},
                    "honesty": {"status": "pass", "score": 2, "hard_gate": True, "evaluator": "fixed_judge"},
                    "instruction_following": {"status": "pass", "score": 2, "hard_gate": False, "evaluator": "fixed_judge"},
                },
            }) for replicate, status, score in ((1, "pass", 2), (2, "pass", 2), (3, "unknown", None))) + "\n", encoding="utf-8")
            output = root / "report.md"
            self.assertEqual(rolebench.main(["report", "--run", str(root), "--output", str(output)]), 0)
            report = output.read_text(encoding="utf-8")
            self.assertIn("| Role | Model | Thinking | Dimension | Source | Pass | Fail | Unknown | Avg known score | Hard gate |", report)
            deterministic_row = "| `task` | `p/a` | `high` | `correctness` | D | 1 | 1 | 0 | 1.00 | yes |"
            judge_row = "| `task` | `p/a` | `high` | `correctness` | J | 2 | 0 | 1 | 2.00 | yes |"
            self.assertIn(deterministic_row, report)
            self.assertIn(judge_row, report)
            self.assertLess(report.index(deterministic_row), report.index(judge_row))
            self.assertIn("| `task` | `p/a` | `high` | `honesty` | D | 0 | 0 | 2 | unknown | yes |", report)
            self.assertIn("| `task` | `p/a` | `high` | `honesty` | J | 3 | 0 | 0 | 2.00 | yes |", report)
            self.assertIn("| `task` | `p/a` | `high` | `instruction_following` | J | 3 | 0 | 0 | 2.00 | no |", report)
            json_output = root / "report.json"
            self.assertEqual(rolebench.main(["report", "--run", str(root), "--format", "json", "--output", str(json_output)]), 0)
            payload = json.loads(json_output.read_text(encoding="utf-8"))
            self.assertEqual(
                [(item["dimension"], item["source_label"]) for item in payload["dimension_summary"]],
                [("correctness", "D"), ("correctness", "J"), ("honesty", "D"), ("honesty", "J"), ("instruction_following", "J")],
            )
            summary = {(item["dimension"], item["source"]): item for item in payload["dimension_summary"]}
            deterministic = summary[("correctness", "deterministic")]
            self.assertEqual((deterministic["pass"], deterministic["fail"], deterministic["unknown"]), (1, 1, 0))
            self.assertEqual(deterministic["average_known_score"], 1.0)
            self.assertTrue(deterministic["hard_gate"])
            judge = summary[("correctness", "fixed_judge")]
            self.assertEqual((judge["pass"], judge["fail"], judge["unknown"]), (2, 0, 1))
            self.assertEqual(judge["average_known_score"], 2.0)
            self.assertEqual(summary[("instruction_following", "fixed_judge")]["hard_gate"], False)

    def test_report_dimensions_table_renders_single_source_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "manifest.json").write_text(json.dumps({"generated_at": "now", "mode": "role-isolated", "roles": ["task"], "assignments": []}), encoding="utf-8")
            (root / "results.jsonl").write_text(json.dumps({"assignment_id": "a", "task_id": "task-1", "replicate": 1, "role": "task", "model": "p/a", "status": "pass", "latency_ms": 5, "output": "done"}) + "\n", encoding="utf-8")
            (root / "evaluations.jsonl").write_text(json.dumps({
                "assignment_id": "a", "task_id": "task-1", "replicate": 1, "role": "task", "model": "p/a",
                "outcome": "pass", "dimensions": {"correctness": {"status": "pass", "score": 2, "evaluator": "deterministic"}},
            }) + "\n", encoding="utf-8")
            output = root / "report.md"
            self.assertEqual(rolebench.main(["report", "--run", str(root), "--output", str(output)]), 0)
            report = output.read_text(encoding="utf-8")
            self.assertIn("| `task` | `p/a` | `-` | `correctness` | D | 1 | 0 | 0 | 2.00 | yes |", report)
            self.assertNotIn("`correctness` | J |", report)

            (root / "evaluations.jsonl").unlink()
            (root / "judge_evaluations.jsonl").write_text(json.dumps({
                "assignment_id": "a", "task_id": "task-1", "replicate": 1, "role": "task", "model": "p/a",
                "outcome": "pass", "dimensions": {"correctness": {"status": "unknown", "score": None, "evaluator": "fixed_judge"}},
            }) + "\n", encoding="utf-8")
            self.assertEqual(rolebench.main(["report", "--run", str(root), "--output", str(output)]), 0)
            report = output.read_text(encoding="utf-8")
            self.assertIn("## Dimensions", report)
            self.assertIn("| `task` | `p/a` | `-` | `correctness` | J | 0 | 0 | 1 | unknown | yes |", report)
            self.assertNotIn("`correctness` | D |", report)

    def test_report_includes_scope_model_dimensions_and_overall_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = {
                "generated_at": "now", "mode": "all-same", "roles": ["task"], "assignments": [],
                "tasks": [{"id": "task-1", "role": "task", "scenario": "repair a local script", "task_type": "code-repair", "workspace": "/tmp/fixture", "difficulty": {"level": 3}, "dimension_targets": ["correctness", "honesty"], "expected": {"output_contains": ["done"]}, "invariants": []}],
            }
            (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            (root / "summary.json").write_text(json.dumps({"status": "completed"}), encoding="utf-8")
            result = {"assignment_id": "a", "task_id": "task-1", "replicate": 1, "role": "task", "model": "p/a", "status": "pass", "latency_ms": 10, "output": "done", "usage": {"cost": {"total": 0.01}}}
            (root / "results.jsonl").write_text(json.dumps(result) + "\n", encoding="utf-8")
            evaluation = {"assignment_id": "a", "task_id": "task-1", "replicate": 1, "role": "task", "model": "p/a", "outcome": "pass", "dimensions": {"correctness": {"status": "pass", "score": 2}}}
            (root / "evaluations.jsonl").write_text(json.dumps(evaluation) + "\n", encoding="utf-8")
            judge = {**evaluation, "dimensions": {"correctness": {"status": "pass", "score": 2}, "honesty": {"status": "unknown", "score": None}}}
            (root / "judge_evaluations.jsonl").write_text(json.dumps(judge) + "\n", encoding="utf-8")
            output = root / "report.md"
            self.assertEqual(rolebench.main(["report", "--run", str(root), "--output", str(output)]), 0)
            report = output.read_text(encoding="utf-8")
            self.assertIn("## Benchmark Scope", report)
            self.assertIn("repair a local script", report)
            self.assertIn("`code-repair`", report)
            self.assertIn("## Model Summary", report)
            self.assertIn("Correctness", report)
            self.assertIn("Honesty", report)
            self.assertIn("Overall", report)
            json_output = root / "report.json"
            self.assertEqual(rolebench.main(["report", "--run", str(root), "--format", "json", "--output", str(json_output)]), 0)
            payload = json.loads(json_output.read_text(encoding="utf-8"))
            self.assertEqual(payload["scope"][0]["task_type"], "code-repair")
            self.assertEqual(payload["model_summary"][0]["dimensions"]["honesty"]["source"], "fixed_judge")

    def test_report_includes_fixed_judge_and_recommendation_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "manifest.json").write_text(json.dumps({"generated_at": "now", "mode": "all-same", "roles": ["task"], "assignments": [], "evaluation": {"fixed_judge": {"enabled": True, "rubric_version": "rolebench-v2-test"}}}), encoding="utf-8")
            (root / "judge_evaluations.jsonl").write_text(json.dumps({"role": "task", "task_id": "task-1", "model": "p/a", "outcome": "pass", "hard_gate_status": "pass", "rubric_version": "rolebench-v2-test", "summary": "supported"}) + "\n", encoding="utf-8")
            (root / "recommendations.json").write_text(json.dumps({"status": "blocked", "roles": [{"role": "task", "model": None, "status": "blocked", "reason": "no eligible candidate"}], "candidates": [{"role": "task", "model": "p/a", "eligible": False, "judge_hard_dimensions": ["correctness"], "blocked_reasons": ["task-2 is incomplete"], "evidence": ["results.jsonl", "judge_evaluations.jsonl"]}]}), encoding="utf-8")
            output = root / "report.md"
            self.assertEqual(rolebench.main(["report", "--run", str(root), "--output", str(output)]), 0)
            report = output.read_text(encoding="utf-8")
            self.assertIn("## Fixed Judge", report)
            self.assertIn("## Recommendations", report)
            self.assertIn("rolebench-v2-test", report)
            self.assertIn("task-2 is incomplete", report)
            self.assertIn("[judge_evaluations.jsonl](judge_evaluations.jsonl)", report)
            json_output = root / "report.json"
            self.assertEqual(rolebench.main(["report", "--run", str(root), "--format", "json", "--output", str(json_output)]), 0)
            payload = json.loads(json_output.read_text(encoding="utf-8"))
            self.assertEqual(payload["recommendations"]["status"], "blocked")
            self.assertEqual(len(payload["judge_evaluations"]), 1)

    def write_conclusions_run(self, root, results=(), evaluations=(), judge_evaluations=(), recommendations=None):
        """Write only the artifacts the Conclusions section reads from."""
        (root / "manifest.json").write_text(json.dumps({"generated_at": "now", "mode": "role-isolated", "roles": ["task"], "assignments": []}), encoding="utf-8")
        for name, rows in (("results.jsonl", results), ("evaluations.jsonl", evaluations), ("judge_evaluations.jsonl", judge_evaluations)):
            if rows:
                (root / name).write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
        if recommendations is not None:
            (root / "recommendations.json").write_text(json.dumps(recommendations), encoding="utf-8")

    def execution_row(self, model, replicate, status="pass", latency_ms=10.0):
        return {
            "assignment_id": f"a-{model}", "task_id": "task-1", "replicate": replicate, "role": "task", "model": model,
            "status": status, "latency_ms": latency_ms, "output": "done",
        }

    def judge_cell(self, model, replicate, outcome, hard_gate_status, dimensions, summary=None):
        return {
            "assignment_id": f"a-{model}", "task_id": "task-1", "replicate": replicate, "role": "task", "model": model,
            "outcome": outcome, "hard_gate_status": hard_gate_status, "summary": summary, "dimensions": dimensions,
        }

    def conclusions_section(self, root):
        output = root / "report.md"
        self.assertEqual(rolebench.main(["report", "--run", str(root), "--output", str(output)]), 0)
        report = output.read_text(encoding="utf-8")
        return report[report.index("## Conclusions"):report.index("## Benchmark Scope")]

    def conclusions_payload(self, root):
        json_output = root / "report.json"
        self.assertEqual(rolebench.main(["report", "--run", str(root), "--format", "json", "--output", str(json_output)]), 0)
        return json.loads(json_output.read_text(encoding="utf-8"))["conclusions"]

    def test_report_conclusions_rank_identities_by_judge_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            passed = {"correctness": {"status": "pass", "score": 2, "hard_gate": True, "evaluator": "fixed_judge"}}
            unknown_correctness = {"correctness": {"status": "unknown", "score": None, "hard_gate": True, "evaluator": "fixed_judge"}}
            unknown_honesty = {"honesty": {"status": "unknown", "score": None, "hard_gate": True, "evaluator": "fixed_judge"}}
            failed_honesty = {"honesty": {"status": "fail", "score": 0, "hard_gate": True, "evaluator": "fixed_judge"}}
            self.write_conclusions_run(
                root,
                results=[self.execution_row(model, replicate) for model in ("p/pass", "p/unknown", "p/fail") for replicate in (1, 2, 3)],
                judge_evaluations=[
                    self.judge_cell("p/pass", 1, "pass", "pass", passed),
                    self.judge_cell("p/pass", 2, "unknown", "unknown", unknown_honesty),
                    self.judge_cell("p/pass", 3, "unknown", "unknown", unknown_correctness),
                    *[self.judge_cell("p/unknown", replicate, "unknown", "unknown", unknown_correctness) for replicate in (1, 2, 3)],
                    self.judge_cell("p/fail", 1, "unknown", "unknown", unknown_correctness),
                    self.judge_cell("p/fail", 2, "unknown", "unknown", unknown_honesty),
                    self.judge_cell("p/fail", 3, "fail", "fail", failed_honesty, summary="asserted unverifiable file:line internals as fact"),
                ],
            )
            conclusions = self.conclusions_section(root)
            payload = self.conclusions_payload(root)
            self.assertEqual(
                payload["ranked"],
                [
                    {"rank": 1, "role": "task", "model": "p/pass", "thinking": None, "reason": "no judge fails; 1 clean pass / 2 unknown"},
                    {"rank": 2, "role": "task", "model": "p/unknown", "thinking": None, "reason": "no judge fails; 3 unknown"},
                    {
                        "rank": 3, "role": "task", "model": "p/fail", "thinking": None,
                        "reason": "judge 1 fail (honesty, hard gate): asserted unverifiable file:line internals as fact",
                    },
                ],
            )
            self.assertEqual(
                payload["lead"],
                [
                    "Top pick: `task` / `p/pass` / `-` — no judge fails; 1 clean pass / 2 unknown",
                    "Evidence: the only identity with a clean judge pass (1 of 3 cells); "
                    "`task` / `p/fail` / `-` failed the judge hard gate (honesty); fewest unresolved cells at rank 1 (2 unknown vs 3).",
                ],
            )
            self.assertIn("1. `task` / `p/pass` / `-` — no judge fails; 1 clean pass / 2 unknown", conclusions.split("\n"))
            self.assertLess(conclusions.index("1. `task` / `p/pass`"), conclusions.index("2. `task` / `p/unknown`"))
            self.assertLess(conclusions.index("2. `task` / `p/unknown`"), conclusions.index("3. `task` / `p/fail`"))
            for banned in ("- Verdict:", "replicate", "Deterministic dimension failures", "### `", "execution "):
                self.assertNotIn(banned, conclusions)

    def test_report_conclusions_weight_hard_gate_fails_above_other_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            unknown = {"correctness": {"status": "unknown", "score": None, "hard_gate": True, "evaluator": "fixed_judge"}}
            soft_fail = {"evidence_quality": {"status": "fail", "score": 0, "hard_gate": False, "evaluator": "fixed_judge", "evidence": ["quoted a file that does not exist"]}}
            hard_fail = {"honesty": {"status": "fail", "score": 0, "hard_gate": True, "evaluator": "fixed_judge"}}
            self.write_conclusions_run(
                root,
                results=[self.execution_row(model, replicate) for model in ("p/clean", "p/soft", "p/hard") for replicate in (1, 2, 3)],
                judge_evaluations=[
                    *[self.judge_cell("p/clean", replicate, "unknown", "unknown", unknown) for replicate in (1, 2, 3)],
                    self.judge_cell("p/soft", 1, "fail", "pass", soft_fail),
                    *[self.judge_cell("p/soft", replicate, "unknown", "unknown", unknown) for replicate in (2, 3)],
                    self.judge_cell("p/hard", 1, "fail", "fail", hard_fail),
                    *[self.judge_cell("p/hard", replicate, "unknown", "unknown", unknown) for replicate in (2, 3)],
                ],
            )
            payload = self.conclusions_payload(root)
            self.assertEqual(
                [(item["rank"], item["model"], item["reason"]) for item in payload["ranked"]],
                [
                    (1, "p/clean", "no judge fails; 3 unknown"),
                    (2, "p/soft", "judge 1 fail (evidence_quality): quoted a file that does not exist"),
                    (3, "p/hard", "judge 1 fail (honesty, hard gate): no deterministic evidence for this dimension"),
                ],
            )
            self.assertEqual(
                payload["lead"],
                [
                    "Top pick: `task` / `p/clean` / `-` — no judge fails; 3 unknown",
                    "Evidence: `task` / `p/soft` / `-` recorded 1 judge fail(s) outside the hard gate; "
                    "`task` / `p/hard` / `-` failed the judge hard gate (honesty).",
                ],
            )

    def test_report_conclusions_prefer_recommendation_status_over_judge_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            unknown = {"correctness": {"status": "unknown", "score": None, "hard_gate": True, "evaluator": "fixed_judge"}}
            self.write_conclusions_run(
                root,
                results=[self.execution_row(model, replicate) for model in ("p/a", "p/b", "p/c") for replicate in (1, 2, 3)],
                judge_evaluations=[
                    self.judge_cell(model, replicate, "unknown", "unknown", unknown)
                    for model in ("p/a", "p/b", "p/c")
                    for replicate in (1, 2, 3)
                ],
                recommendations={
                    "status": "complete",
                    "roles": [{"role": "task", "model": "p/b", "thinking": None, "status": "recommended", "reason": "hard gates passed; highest fixed-judge quality score"}],
                    "candidates": [
                        {"role": "task", "model": "p/b", "thinking": None, "eligible": True, "quality_score": 1.75, "blocked_reasons": []},
                        {"role": "task", "model": "p/c", "thinking": None, "eligible": True, "quality_score": 1.55, "blocked_reasons": []},
                        {"role": "task", "model": "p/a", "thinking": None, "eligible": False, "quality_score": 1.25, "blocked_reasons": ["fixed judge hard gate failed"]},
                    ],
                },
            )
            payload = self.conclusions_payload(root)
            self.assertEqual(
                [(item["rank"], item["model"], item["reason"]) for item in payload["ranked"]],
                [
                    (1, "p/b", "`recommended` (quality 1.75) — hard gates passed; highest fixed-judge quality score; no judge fails; 3 unknown"),
                    (2, "p/c", "`eligible` (quality 1.55) — eligible; not the top-ranked candidate for this role; no judge fails; 3 unknown"),
                    (3, "p/a", "`blocked` (quality 1.25) — fixed judge hard gate failed; no judge fails; 3 unknown"),
                ],
            )
            self.assertEqual(
                payload["lead"],
                [
                    "Top pick: `task` / `p/b` / `-` — `recommended` (quality 1.75) — hard gates passed; highest fixed-judge quality score; no judge fails; 3 unknown",
                    "Evidence: every identity shares the same judge record (no judge fails; 3 unknown), so only the recommendation status, execution errors, and latency separate them.",
                ],
            )

    def test_report_conclusions_break_ties_by_errors_then_latency(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            unknown = {"correctness": {"status": "unknown", "score": None, "hard_gate": True, "evaluator": "fixed_judge"}}
            self.write_conclusions_run(
                root,
                results=[
                    self.execution_row("p/errored", 1, status="error", latency_ms=5.0),
                    self.execution_row("p/errored", 2, latency_ms=5.0),
                    self.execution_row("p/errored", 3, latency_ms=5.0),
                    *[self.execution_row("p/slow", replicate, latency_ms=50.0) for replicate in (1, 2, 3)],
                    *[self.execution_row("p/quick", replicate, latency_ms=10.0) for replicate in (1, 2, 3)],
                ],
                judge_evaluations=[
                    self.judge_cell(model, replicate, "unknown", "unknown", unknown)
                    for model in ("p/errored", "p/slow", "p/quick")
                    for replicate in (1, 2, 3)
                ],
            )
            payload = self.conclusions_payload(root)
            self.assertEqual([item["model"] for item in payload["ranked"]], ["p/quick", "p/slow", "p/errored"])
            self.assertEqual(
                payload["lead"],
                [
                    "No identity is distinguishable on judge evidence: all 3 identities share the same record "
                    "(no judge fails; 3 unknown); rank order follows execution errors, then p50 latency."
                ],
            )

    def test_report_conclusions_label_cost_and_latency_advantages_separately(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            passed = {"correctness": {"status": "pass", "score": 2, "hard_gate": True, "evaluator": "fixed_judge"}}
            unknown = {"correctness": {"status": "unknown", "score": None, "hard_gate": True, "evaluator": "fixed_judge"}}

            def lead_for(top_latency, top_cost, rival_latency, rival_cost):
                self.write_conclusions_run(
                    root,
                    results=[
                        {**self.execution_row("p/top", 1, latency_ms=top_latency), "usage": {"cost": {"total": top_cost}}},
                        {**self.execution_row("p/rival", 1, latency_ms=rival_latency), "usage": {"cost": {"total": rival_cost}}},
                    ],
                    judge_evaluations=[
                        self.judge_cell("p/top", 1, "pass", "pass", passed),
                        self.judge_cell("p/rival", 1, "unknown", "unknown", unknown),
                    ],
                )
                return self.conclusions_payload(root)["lead"]

            cost_only = lead_for(50.0, 0.01, 10.0, 0.02)
            self.assertEqual(cost_only[-1], "Cost: lowest observed cost ($0.010000 vs $0.020000).")
            self.assertEqual(len(cost_only), 3)
            self.assertNotIn("latency", " ".join(cost_only))
            self.assertNotIn("Latency", " ".join(cost_only))

            latency_only = lead_for(10.0, 0.02, 50.0, 0.01)
            self.assertEqual(latency_only[-1], "Latency: fastest p50 latency (10.0 ms vs 50.0 ms).")
            self.assertNotIn("Cost", " ".join(latency_only))
            self.assertNotIn("observed cost", " ".join(latency_only))

            both = lead_for(10.0, 0.01, 50.0, 0.02)
            self.assertEqual(
                both[-1],
                "Cost and latency: lowest observed cost ($0.010000 vs $0.020000); fastest p50 latency (10.0 ms vs 50.0 ms).",
            )

    def test_report_conclusions_prefer_judge_summary_over_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.write_conclusions_run(
                root,
                judge_evaluations=[
                    self.judge_cell(
                        "p/a", 1, "fail", "fail",
                        {"correctness": {"status": "fail", "score": 0, "hard_gate": True, "evaluator": "fixed_judge", "evidence": ["no verification was run"]}},
                        summary="Timeout produced no verifiable output",
                    ),
                ],
            )
            conclusions = self.conclusions_section(root)
            self.assertIn(
                "1. `task` / `p/a` / `-` — judge 1 fail (correctness, hard gate): Timeout produced no verifiable output",
                conclusions,
            )
            self.assertNotIn("no verification was run", conclusions)
            self.assertEqual(
                self.conclusions_payload(root)["lead"],
                ["Top pick: `task` / `p/a` / `-` — judge 1 fail (correctness, hard gate): Timeout produced no verifiable output"],
            )

    def test_report_conclusions_render_without_judge_data(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            passed = {"correctness": {"status": "pass", "score": 2, "evaluator": "deterministic"}}
            self.write_conclusions_run(
                root,
                results=[self.execution_row("p/a", 1)],
                evaluations=[{"assignment_id": "a-p/a", "task_id": "task-1", "replicate": 1, "role": "task", "model": "p/a", "outcome": "pass", "dimensions": passed}],
            )
            conclusions = self.conclusions_section(root)
            self.assertIn("1. `task` / `p/a` / `-` — no deterministic fails; 1 clean pass", conclusions)
            self.assertNotIn("judge", conclusions)
            self.assertEqual(
                self.conclusions_payload(root),
                {
                    "lead": ["Top pick: `task` / `p/a` / `-` — no deterministic fails; 1 clean pass"],
                    "ranked": [{"rank": 1, "role": "task", "model": "p/a", "thinking": None, "reason": "no deterministic fails; 1 clean pass"}],
                },
            )

            failed = {"correctness": {"status": "fail", "score": 0, "hard_gate": True, "evaluator": "deterministic", "evidence": ["the script still exits non-zero"]}}
            self.write_conclusions_run(
                root,
                results=[self.execution_row("p/a", 1)],
                evaluations=[{"assignment_id": "a-p/a", "task_id": "task-1", "replicate": 1, "role": "task", "model": "p/a", "outcome": "fail", "hard_gate_status": "fail", "dimensions": failed}],
            )
            self.assertIn(
                "1. `task` / `p/a` / `-` — deterministic 1 fail (correctness, hard gate): the script still exits non-zero",
                self.conclusions_section(root),
            )

            for name in ("results.jsonl", "evaluations.jsonl"):
                (root / name).unlink()
            output = root / "report.md"
            self.assertEqual(rolebench.main(["report", "--run", str(root), "--output", str(output)]), 0)
            self.assertIn(
                "## Conclusions\n\nNo execution, evaluation, or fixed-judge rows were recorded for this run.\n\n## Benchmark Scope",
                output.read_text(encoding="utf-8"),
            )
            self.assertEqual(self.conclusions_payload(root), {"lead": [], "ranked": []})


    def test_report_expands_task_dimension_and_output_details(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prompt = "Repair the fixture script.\nReport what remains uncertain."
            long_output = "line one\n~~~~\nnested fence attempt\n" + "x" * 2500
            manifest = {
                "generated_at": "now", "mode": "all-same", "roles": ["task"], "assignments": [],
                "tasks": [{
                    "id": "task-1", "role": "task", "scenario": "repair a local script", "task_type": "code-repair",
                    "prompt": prompt, "workspace": "/tmp/fixture", "difficulty": {"level": 3},
                    "dimension_targets": ["correctness", "honesty"],
                    "expected": {"output_contains": ["done"]},
                    "invariants": [{"id": "patch", "type": "file_contains", "path": "fix.py", "patterns": ["PASS"]}],
                    "failure_mode": {"required_output_regex": ["(?i)uncertain"]},
                }],
            }
            (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            rows = [
                {"assignment_id": "a", "task_id": "task-1", "replicate": 1, "role": "task", "model": "p/a", "thinking": "high", "status": "pass", "execution_status": "completed", "latency_ms": 12.5, "output": long_output, "usage": {"cost": {"total": 0.02}}},
                {"assignment_id": "a", "task_id": "task-1", "replicate": 3, "role": "task", "model": "p/a", "thinking": "high", "status": "error", "execution_status": "failed", "latency_ms": 4.0, "output": None, "usage": {}},
            ]
            (root / "results.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
            evaluation = {
                "assignment_id": "a", "task_id": "task-1", "replicate": 1, "role": "task", "model": "p/a", "thinking": "high",
                "outcome": "fail", "hard_gate_status": "fail",
                "dimensions": {
                    "correctness": {"status": "fail", "score": 0, "evidence": ["output is missing the required marker"], "evaluator": "deterministic"},
                    "honesty": {"status": "unknown", "score": None, "evidence": ["no deterministic evidence for this dimension"], "evaluator": "none"},
                },
            }
            (root / "evaluations.jsonl").write_text(json.dumps(evaluation) + "\n", encoding="utf-8")
            judge = {
                "assignment_id": "a", "task_id": "task-1", "replicate": 1, "role": "task", "model": "p/a", "thinking": "high",
                "outcome": "pass", "hard_gate_status": "pass", "rubric_version": "rolebench-v2-test",
                "summary": "judge summary text",
                "dimensions": {
                    "correctness": {"status": "pass", "score": 2, "evidence": ["matched the declared contract"], "evaluator": "fixed_judge", "hard_gate": True},
                    "honesty": {"status": "unknown", "score": None, "evidence": ["judge could not verify the claim"], "evaluator": "fixed_judge", "hard_gate": True},
                },
            }
            (root / "judge_evaluations.jsonl").write_text(json.dumps(judge) + "\n", encoding="utf-8")
            output = root / "report.md"
            self.assertEqual(rolebench.main(["report", "--run", str(root), "--output", str(output)]), 0)
            report = output.read_text(encoding="utf-8")
            self.assertIn("## Task Details", report)
            self.assertIn(prompt, report)
            self.assertIn("`code-repair`", report)
            self.assertIn("`/tmp/fixture`", report)
            self.assertIn('"output_contains"', report)
            self.assertIn("Failure mode:", report)
            self.assertIn("## Dimension Details", report)
            self.assertIn("- D replicate 1: `fail` (score 0, evaluator `deterministic`)", report)
            self.assertIn("- J replicate 1: `pass` (score 2, evaluator `fixed_judge`)", report)
            self.assertIn("  - matched the declared contract", report)
            self.assertIn("  - judge could not verify the claim", report)
            self.assertNotIn("no deterministic evidence for this dimension", report)
            self.assertIn("judge summary text", report)
            self.assertIn("## Outputs", report)
            self.assertIn(f"- Cost: $0.020000", report)
            self.assertIn(f"[truncated at 2000 of {len(long_output)} characters]", report)
            self.assertIn("\n~~~~~\n", report)
            self.assertIn("~~~~\nnested fence attempt", report)
            self.assertIn("`task-1` / `p/a` / replicate 3: no output captured", report)
            json_output = root / "report.json"
            self.assertEqual(rolebench.main(["report", "--run", str(root), "--format", "json", "--output", str(json_output)]), 0)
            payload = json.loads(json_output.read_text(encoding="utf-8"))
            self.assertEqual(payload["task_details"][0]["prompt"], prompt)
            self.assertEqual(payload["task_details"][0]["failure_mode"], {"required_output_regex": ["(?i)uncertain"]})
            dimensions = {item["dimension"]: item for item in payload["dimension_details"]}
            self.assertEqual(dimensions["correctness"]["deterministic"][0]["status"], "fail")
            self.assertEqual(dimensions["correctness"]["fixed_judge"][0]["evidence"], ["matched the declared contract"])
            self.assertEqual(dimensions["honesty"]["fixed_judge"][0]["status"], "unknown")
            self.assertEqual([entry["replicate"] for entry in payload["outputs"]], [1, 3])
            self.assertEqual(payload["outputs"][0]["output"], long_output)
            self.assertEqual(payload["outputs"][0]["cost_usd"], 0.02)
            self.assertIsNone(payload["outputs"][1]["output"])
            self.assertIsNone(payload["outputs"][1]["cost_usd"])

    def test_report_omits_detail_sections_when_data_is_absent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "manifest.json").write_text(json.dumps({"generated_at": "now", "mode": "role-isolated", "roles": ["task"], "assignments": []}), encoding="utf-8")
            output = root / "report.md"
            self.assertEqual(rolebench.main(["report", "--run", str(root), "--output", str(output)]), 0)
            report = output.read_text(encoding="utf-8")
            self.assertNotIn("## Task Details", report)
            self.assertNotIn("## Dimension Details", report)
            self.assertNotIn("## Outputs", report)
            json_output = root / "report.json"
            self.assertEqual(rolebench.main(["report", "--run", str(root), "--format", "json", "--output", str(json_output)]), 0)
            payload = json.loads(json_output.read_text(encoding="utf-8"))
            self.assertEqual(payload["task_details"], [])
            self.assertEqual(payload["dimension_details"], [])
            self.assertEqual(payload["outputs"], [])

    def test_report_hides_dimensions_without_recorded_evidence(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = {
                "generated_at": "now", "mode": "role-isolated", "roles": ["task"], "assignments": [],
                "tasks": [{"id": "task-1", "role": "task", "prompt": "do it", "dimension_targets": ["correctness"]}],
            }
            (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            (root / "results.jsonl").write_text(json.dumps({"assignment_id": "a", "task_id": "task-1", "replicate": 1, "role": "task", "model": "p/a", "status": "pass", "latency_ms": 5, "output": "done"}) + "\n", encoding="utf-8")
            evaluation = {
                "assignment_id": "a", "task_id": "task-1", "replicate": 1, "role": "task", "model": "p/a",
                "outcome": "unknown", "hard_gate_status": "unknown",
                "dimensions": {"correctness": {"status": "unknown", "score": None, "evidence": ["no deterministic evidence for this dimension"], "evaluator": "none"}},
            }
            (root / "evaluations.jsonl").write_text(json.dumps(evaluation) + "\n", encoding="utf-8")
            output = root / "report.md"
            self.assertEqual(rolebench.main(["report", "--run", str(root), "--output", str(output)]), 0)
            report = output.read_text(encoding="utf-8")
            self.assertIn("## Task Details", report)
            self.assertIn("do it", report)
            self.assertNotIn("## Dimension Details", report)
            self.assertIn("## Outputs", report)
            self.assertIn("~~~~\ndone\n~~~~", report)
            json_output = root / "report.json"
            self.assertEqual(rolebench.main(["report", "--run", str(root), "--format", "json", "--output", str(json_output)]), 0)
            payload = json.loads(json_output.read_text(encoding="utf-8"))
            self.assertEqual(payload["dimension_details"], [])
            self.assertEqual(payload["outputs"][0]["output"], "done")

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
            summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["reports"]["status"], "generated")
            self.assertEqual(summary["reports"]["markdown"], str(output / "report.md"))
            report = (output / "report.md").read_text(encoding="utf-8")
            self.assertIn("## Benchmark Scope", report)
            self.assertIn("## Model Summary", report)
            self.assertIn("Correctness", report)
            report_json = json.loads((output / "report.json").read_text(encoding="utf-8"))
            self.assertEqual(report_json["scope"][0]["task_id"], "task-1")
            self.assertIn("correctness", report_json["model_summary"][0]["dimensions"])

    @patch.object(rolebench, "report_command", side_effect=OSError("disk full"))
    def test_report_failure_is_recorded_without_reclassifying_run(self, report):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = self.make_run_args(root, execute=False, confirm=False)
            with patch.object(rolebench, "load_tasks", return_value=[{"id": "task-1", "role": "task", "prompt": "do", "dimension_targets": ["correctness"]}]):
                self.assertEqual(rolebench.run_command(args), 4)
            summary = json.loads((root / "run" / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["status"], "preview")
            self.assertIsNone(summary["stopped_reason"])
            self.assertEqual(summary["reports"]["status"], "failed")
            self.assertIn("disk full", summary["reports"]["error"])

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

    def test_parallel_direct_cells_respect_global_jobs_and_persist_unique_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lock = threading.Lock()
            gate = threading.Barrier(3)
            active = 0
            peak = 0

            def candidate(*_args):
                nonlocal active, peak
                with lock:
                    active += 1
                    peak = max(peak, active)
                gate.wait(timeout=2)
                time.sleep(0.02)
                with lock:
                    active -= 1
                return {"status": "pass", "execution_status": "completed", "provider_call_started": False, "output": "done"}

            task = {"id": "task-1", "role": "task", "prompt": "do", "dimension_targets": ["correctness"]}
            with patch.object(rolebench, "load_tasks", return_value=[task]):
                with patch.object(rolebench, "run_one", side_effect=candidate) as run:
                    self.assertEqual(rolebench.run_command(self.make_run_args(root, runs=3, jobs=3)), 0)
            rows = rolebench.load_jsonl(root / "run" / "results.jsonl", "results.jsonl")
            identities = {(row["assignment_id"], row["task_id"], row["replicate"]) for row in rows}
            summary = json.loads((root / "run" / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(run.call_count, 3)
            self.assertEqual(peak, 3)
            self.assertEqual(len(rows), 3)
            self.assertEqual(len(identities), 3)
            self.assertEqual(summary["jobs"], 3)
            self.assertEqual(summary["effective_jobs"], 3)
            self.assertIn("Latency mode: `concurrent`", (root / "run" / "report.md").read_text(encoding="utf-8"))

    def test_parallel_schedule_is_seed_stable_and_round_robins_assignments(self):
        assignments = [
            {"id": "a", "mode": "role-isolated", "role": "task", "mapping": {"task": "p/a"}},
            {"id": "b", "mode": "role-isolated", "role": "task", "mapping": {"task": "p/b"}},
        ]
        tasks = [{"id": "task-1", "role": "task"}]
        first = rolebench.build_cell_schedule(assignments, tasks, 3, seed=7, jobs=2)
        second = rolebench.build_cell_schedule(assignments, tasks, 3, seed=7, jobs=2)
        ids = [cell["assignment"]["id"] for cell in first]
        self.assertEqual(ids, [cell["assignment"]["id"] for cell in second])
        self.assertNotEqual(ids[0], ids[1])
        self.assertEqual(ids[::2], [ids[0]] * 3)
        self.assertEqual(ids[1::2], [ids[1]] * 3)

    def test_parallel_judge_starts_only_after_candidate_result_is_persisted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pricing = root / "pricing.json"
            pricing.write_text(json.dumps({
                "p/a": {"input_per_million": 1, "output_per_million": 1},
                "judge/model": {"input_per_million": 1, "output_per_million": 1},
            }), encoding="utf-8")
            persisted_before_judge = []

            def judge(_omp_bin, judge_model, task, result, _evaluation, _timeout):
                rows = rolebench.load_jsonl(root / "run" / "results.jsonl", "results.jsonl")
                key = (result["assignment_id"], result["task_id"], result["replicate"])
                persisted_before_judge.append(any(rolebench.cell_key(row) == key for row in rows))
                raw = {
                    "assignment_id": result["assignment_id"],
                    "task_id": task["id"],
                    "replicate": result["replicate"],
                    "role": task["role"],
                    "model": judge_model,
                    "provider_call_started": False,
                    "status": "pass",
                    "execution_status": "completed",
                }
                evaluation = {
                    "assignment_id": result["assignment_id"],
                    "task_id": task["id"],
                    "replicate": result["replicate"],
                    "role": task["role"],
                    "model": result["model"],
                    "execution_status": "completed",
                    "outcome": "pass",
                    "hard_gate_status": "pass",
                    "checks": [],
                    "dimensions": {},
                }
                return raw, evaluation

            task = {"id": "task-1", "role": "task", "prompt": "do", "dimension_targets": ["correctness"]}
            candidate = {"status": "pass", "execution_status": "completed", "provider_call_started": False, "output": "done"}
            args = self.make_run_args(root, runs=2, jobs=2, pricing_file=str(pricing), judge_model="judge/model")
            with patch.object(rolebench, "load_tasks", return_value=[task]):
                with patch.object(rolebench, "run_one", return_value=candidate):
                    with patch.object(rolebench, "run_fixed_judge", side_effect=judge):
                        self.assertEqual(rolebench.run_command(args), 0)
            self.assertEqual(persisted_before_judge, [True, True])
            self.assertEqual(len(rolebench.load_jsonl(root / "run" / "judge_results.jsonl", "judge_results.jsonl")), 2)

    def test_jobs_one_preserves_candidate_then_judge_order_per_cell(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pricing = root / "pricing.json"
            pricing.write_text(json.dumps({
                "p/a": {"input_per_million": 1, "output_per_million": 1},
                "judge/model": {"input_per_million": 1, "output_per_million": 1},
            }), encoding="utf-8")
            calls = []

            def candidate(_omp_bin, _assignment, _task, replicate, *_args):
                calls.append(("candidate", replicate))
                return {"status": "pass", "execution_status": "completed", "provider_call_started": False, "output": "done"}

            def judge(_omp_bin, judge_model, task, result, _evaluation, _timeout):
                calls.append(("judge", result["replicate"]))
                raw = {
                    "schema_version": rolebench.FIXED_JUDGE_SCHEMA_VERSION,
                    "judge_result_id": f"{result['assignment_id']}:{task['id']}:{result['replicate']}",
                    "assignment_id": result["assignment_id"],
                    "task_id": task["id"],
                    "role": task["role"],
                    "model": judge_model,
                    "thinking": None,
                    "replicate": result["replicate"],
                    "evaluator": "fixed_judge",
                    "rubric_version": rolebench.FIXED_JUDGE_RUBRIC_VERSION,
                    "provider_call_started": False,
                    "status": "pass",
                    "execution_status": "completed",
                }
                evaluation = {
                    "assignment_id": result["assignment_id"],
                    "task_id": task["id"],
                    "replicate": result["replicate"],
                    "role": task["role"],
                    "model": result["model"],
                    "execution_status": "completed",
                    "outcome": "pass",
                    "hard_gate_status": "pass",
                    "checks": [],
                    "dimensions": {},
                }
                return raw, evaluation

            task = {"id": "task-1", "role": "task", "prompt": "do", "dimension_targets": ["correctness"]}
            args = self.make_run_args(root, runs=2, jobs=1, pricing_file=str(pricing), judge_model="judge/model")
            with patch.object(rolebench, "load_tasks", return_value=[task]):
                with patch.object(rolebench, "run_one", side_effect=candidate):
                    with patch.object(rolebench, "run_fixed_judge", side_effect=judge):
                        self.assertEqual(rolebench.run_command(args), 0)
            self.assertEqual(calls, [("candidate", 1), ("judge", 1), ("candidate", 2), ("judge", 2)])

    def test_parallel_budget_stop_drains_only_already_in_flight_cells(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pricing = root / "pricing.json"
            pricing.write_text(json.dumps({"p/a": {"input_per_million": 1, "output_per_million": 1}}), encoding="utf-8")
            gate = threading.Barrier(2)

            def candidate(*_args):
                gate.wait(timeout=2)
                return {
                    "status": "pass",
                    "execution_status": "completed",
                    "provider_call_started": True,
                    "output": "done",
                    "usage": {"cost": {"total": 0.6}},
                }

            task = {"id": "task-1", "role": "task", "prompt": "do"}
            args = self.make_run_args(root, runs=4, jobs=2, pricing_file=str(pricing), max_cost_usd=0.5)
            with patch.object(rolebench, "load_tasks", return_value=[task]):
                with patch.object(rolebench, "run_one", side_effect=candidate) as run:
                    self.assertEqual(rolebench.run_command(args), 0)
            summary = json.loads((root / "run" / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(run.call_count, 2)
            self.assertEqual(summary["result_rows"], 2)
            self.assertEqual(summary["budget_status"], "reached")
            self.assertEqual(summary["actual_cost_usd"], 1.2)

    def test_parallel_execution_rejects_delegated_and_resume_job_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(rolebench.RoleBenchError, "only with --executor direct"):
                rolebench.run_command(self.make_run_args(root, jobs=2, executor="delegated"))
            task = {"id": "task-1", "role": "task", "prompt": "do"}
            row = {"status": "pass", "execution_status": "completed", "provider_call_started": False, "output": "done"}
            with patch.object(rolebench, "load_tasks", return_value=[task]):
                with patch.object(rolebench, "run_one", return_value=row):
                    self.assertEqual(rolebench.run_command(self.make_run_args(root)), 0)
            with patch.object(rolebench, "load_tasks", return_value=[task]):
                with self.assertRaisesRegex(rolebench.RoleBenchError, "experiment does not match"):
                    rolebench.run_command(self.make_run_args(root, jobs=2, resume=True))

    def test_resume_rejects_audit_rows_outside_expected_cell_matrix(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            task = {"id": "task-1", "role": "task", "prompt": "do"}
            row = {"status": "pass", "execution_status": "completed", "provider_call_started": False, "output": "done"}
            with patch.object(rolebench, "load_tasks", return_value=[task]):
                with patch.object(rolebench, "run_one", return_value=row):
                    self.assertEqual(rolebench.run_command(self.make_run_args(root)), 0)
            results_path = root / "run" / "results.jsonl"
            persisted = json.loads(results_path.read_text(encoding="utf-8"))
            persisted["assignment_id"] = "unexpected-assignment"
            results_path.write_text(json.dumps(persisted) + "\n", encoding="utf-8")
            with patch.object(rolebench, "load_tasks", return_value=[task]):
                with self.assertRaisesRegex(rolebench.RoleBenchError, "do not belong to the expected matrix"):
                    rolebench.run_command(self.make_run_args(root, resume=True))

    def test_resume_rejects_candidate_metadata_from_another_assignment(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            task = {"id": "task-1", "role": "task", "prompt": "do"}
            row = {"status": "pass", "execution_status": "completed", "provider_call_started": False, "output": "done"}
            with patch.object(rolebench, "load_tasks", return_value=[task]):
                with patch.object(rolebench, "run_one", return_value=row):
                    self.assertEqual(rolebench.run_command(self.make_run_args(root)), 0)
            results_path = root / "run" / "results.jsonl"
            persisted = json.loads(results_path.read_text(encoding="utf-8"))
            persisted["model"] = "p/wrong"
            results_path.write_text(json.dumps(persisted) + "\n", encoding="utf-8")
            with patch.object(rolebench, "load_tasks", return_value=[task]):
                with self.assertRaisesRegex(rolebench.RoleBenchError, "candidate result row metadata"):
                    rolebench.run_command(self.make_run_args(root, resume=True))

    def test_resume_rejects_tampered_deterministic_evaluation_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            task = {"id": "task-1", "role": "task", "prompt": "do", "dimension_targets": ["correctness"]}
            evaluation = {
                "schema_version": 1,
                "evaluation_status": "completed",
                "assignment_id": "assignment-001-01",
                "task_id": "task-1",
                "replicate": 1,
                "role": "task",
                "model": "p/a",
                "thinking": None,
                "execution_status": "completed",
                "outcome": "pass",
                "hard_gate_status": "pass",
                "checks": [],
                "dimensions": {},
            }
            row = {"status": "pass", "execution_status": "completed", "provider_call_started": False, "output": "done", "_evaluation": evaluation}
            with patch.object(rolebench, "load_tasks", return_value=[task]):
                with patch.object(rolebench, "run_one", return_value=row):
                    self.assertEqual(rolebench.run_command(self.make_run_args(root)), 0)
            path = root / "run" / "evaluations.jsonl"
            persisted = json.loads(path.read_text(encoding="utf-8"))
            persisted["model"] = "stale/model"
            path.write_text(json.dumps(persisted) + "\n", encoding="utf-8")
            with patch.object(rolebench, "load_tasks", return_value=[task]):
                with self.assertRaisesRegex(rolebench.RoleBenchError, "deterministic evaluation row metadata"):
                    rolebench.run_command(self.make_run_args(root, resume=True))

    def test_resume_rejects_tampered_raw_result_when_evaluation_is_missing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            task = {"id": "task-1", "role": "task", "prompt": "do", "dimension_targets": ["correctness"]}
            evaluation = {
                "schema_version": 1,
                "evaluation_status": "completed",
                "assignment_id": "assignment-001-01",
                "task_id": "task-1",
                "replicate": 1,
                "role": "task",
                "model": "p/a",
                "thinking": None,
                "execution_status": "completed",
                "outcome": "pass",
                "hard_gate_status": "pass",
                "checks": [],
                "dimensions": {},
            }
            row = {"status": "pass", "execution_status": "completed", "provider_call_started": False, "output": "done", "_evaluation": evaluation}
            with patch.object(rolebench, "load_tasks", return_value=[task]):
                with patch.object(rolebench, "run_one", return_value=row):
                    self.assertEqual(rolebench.run_command(self.make_run_args(root)), 0)
            output = root / "run"
            (output / "evaluations.jsonl").unlink()
            path = output / "results.jsonl"
            persisted = json.loads(path.read_text(encoding="utf-8"))
            persisted["output"] = "tampered"
            persisted["usage"] = {"cost": {"total": 999}}
            path.write_text(json.dumps(persisted) + "\n", encoding="utf-8")
            manifest_before = (output / "manifest.json").read_bytes()
            with patch.object(rolebench, "load_tasks", return_value=[task]):
                with patch.object(rolebench, "run_one") as run:
                    with self.assertRaisesRegex(rolebench.RoleBenchError, "candidate result row content digest"):
                        rolebench.run_command(self.make_run_args(root, resume=True))
            run.assert_not_called()
            self.assertEqual((output / "manifest.json").read_bytes(), manifest_before)

    def test_resume_rejects_changed_source_workspace_before_provider_call(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "fixture"
            source.mkdir()
            (source / "input.txt").write_text("before", encoding="utf-8")
            task = {"id": "task-1", "role": "task", "prompt": "do", "workspace": str(source)}
            row = {"status": "pass", "execution_status": "completed", "provider_call_started": False, "output": "done"}
            with patch.object(rolebench, "load_tasks", return_value=[task]):
                with patch.object(rolebench, "run_one", return_value=row):
                    self.assertEqual(rolebench.run_command(self.make_run_args(root)), 0)
            (source / "input.txt").write_text("changed", encoding="utf-8")
            with patch.object(rolebench, "load_tasks", return_value=[task]):
                with patch.object(rolebench, "run_one") as run:
                    with self.assertRaisesRegex(rolebench.RoleBenchError, "source workspace or image content differs"):
                        rolebench.run_command(self.make_run_args(root, resume=True))
            run.assert_not_called()

    def test_resume_source_fingerprint_ignores_nested_output_and_follows_directory_symlink(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "fixture"
            source.mkdir()
            target = root / "linked"
            target.mkdir()
            (target / "input.txt").write_text("before", encoding="utf-8")
            (source / "linked").symlink_to(target, target_is_directory=True)
            output = source / "run"
            output.mkdir()
            (source / "output-alias").symlink_to(output, target_is_directory=True)
            task = {"id": "task-1", "role": "task", "prompt": "do", "workspace": str(source)}
            row = {"status": "pass", "execution_status": "completed", "provider_call_started": False, "output": "done"}
            args = self.make_run_args(root, output=str(output))
            with patch.object(rolebench, "load_tasks", return_value=[task]):
                with patch.object(rolebench, "run_one", return_value=row):
                    self.assertEqual(rolebench.run_command(args), 0)
            (target / "input.txt").write_text("changed", encoding="utf-8")
            with patch.object(rolebench, "load_tasks", return_value=[task]):
                with patch.object(rolebench, "run_one") as run:
                    with self.assertRaisesRegex(rolebench.RoleBenchError, "source workspace or image content differs"):
                        rolebench.run_command(self.make_run_args(root, output=str(output), resume=True))
            run.assert_not_called()

    def test_resume_rejects_stale_judge_model_and_rubric(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pricing = root / "pricing.json"
            pricing.write_text(json.dumps({
                "p/a": {"input_per_million": 1, "output_per_million": 1},
                "judge/model": {"input_per_million": 1, "output_per_million": 1},
            }), encoding="utf-8")
            task = {"id": "task-1", "role": "task", "prompt": "do", "dimension_targets": ["correctness"]}
            candidate = {"status": "pass", "execution_status": "completed", "provider_call_started": False, "output": "done"}
            judge_row = {
                "schema_version": rolebench.FIXED_JUDGE_SCHEMA_VERSION,
                "judge_result_id": "assignment-001-01:task-1:1",
                "assignment_id": "assignment-001-01",
                "task_id": "task-1",
                "role": "task",
                "model": "judge/model",
                "thinking": None,
                "replicate": 1,
                "evaluator": "fixed_judge",
                "rubric_version": rolebench.FIXED_JUDGE_RUBRIC_VERSION,
                "provider_call_started": False,
                "status": "pass",
                "execution_status": "completed",
                "output": '{"dimensions":{"correctness":{"status":"pass","score":2,"evidence":"ok"}}}',
            }
            judge_evaluation = {
                "assignment_id": "assignment-001-01",
                "task_id": "task-1",
                "replicate": 1,
                "role": "task",
                "model": "p/a",
                "execution_status": "completed",
                "outcome": "pass",
                "hard_gate_status": "pass",
                "dimensions": {"correctness": {"status": "pass", "score": 2, "evidence": "ok", "evaluator": "fixed_judge"}},
            }
            args = self.make_run_args(root, pricing_file=str(pricing), judge_model="judge/model")
            with patch.object(rolebench, "load_tasks", return_value=[task]):
                with patch.object(rolebench, "run_one", return_value=candidate):
                    with patch.object(rolebench, "run_fixed_judge", return_value=(judge_row, judge_evaluation)):
                        self.assertEqual(rolebench.run_command(args), 0)
            judge_path = root / "run" / "judge_results.jsonl"
            persisted = json.loads(judge_path.read_text(encoding="utf-8"))
            persisted["model"] = "judge/wrong"
            persisted["rubric_version"] = "stale-rubric"
            judge_path.write_text(json.dumps(persisted) + "\n", encoding="utf-8")
            with patch.object(rolebench, "load_tasks", return_value=[task]):
                with self.assertRaisesRegex(rolebench.RoleBenchError, "fixed-judge result row metadata"):
                    rolebench.run_command(self.make_run_args(root, pricing_file=str(pricing), judge_model="judge/model", resume=True))

    def test_resume_rejects_judge_evaluation_that_disagrees_with_raw_response(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pricing = root / "pricing.json"
            pricing.write_text(json.dumps({
                "p/a": {"input_per_million": 1, "output_per_million": 1},
                "judge/model": {"input_per_million": 1, "output_per_million": 1},
            }), encoding="utf-8")
            task = {"id": "task-1", "role": "task", "prompt": "do", "dimension_targets": ["correctness"]}
            candidate_evaluation = {
                "schema_version": 1,
                "evaluation_status": "completed",
                "assignment_id": "assignment-001-01",
                "task_id": "task-1",
                "replicate": 1,
                "role": "task",
                "model": "p/a",
                "thinking": None,
                "execution_status": "completed",
                "outcome": "pass",
                "hard_gate_status": "pass",
                "checks": [],
                "dimensions": {},
            }
            candidate = {"status": "pass", "execution_status": "completed", "provider_call_started": False, "output": "done", "_evaluation": candidate_evaluation}

            def judge(_omp_bin, judge_model, fixture, result, _evaluation, _timeout):
                dimensions = {"correctness": {"status": "pass", "score": 2, "evidence": ["ok"]}}
                raw = {
                    "schema_version": rolebench.FIXED_JUDGE_SCHEMA_VERSION,
                    "judge_result_id": f"{result['assignment_id']}:{result['task_id']}:{result['replicate']}",
                    "assignment_id": result["assignment_id"],
                    "task_id": result["task_id"],
                    "role": result["role"],
                    "model": judge_model,
                    "thinking": None,
                    "replicate": result["replicate"],
                    "evaluator": "fixed_judge",
                    "rubric_version": rolebench.FIXED_JUDGE_RUBRIC_VERSION,
                    "provider_call_started": False,
                    "status": "pass",
                    "execution_status": "completed",
                    "output": json.dumps({"dimensions": dimensions}),
                }
                return raw, rolebench.build_judge_evaluation(fixture, result, raw, dimensions)

            args = self.make_run_args(root, pricing_file=str(pricing), judge_model="judge/model")
            with patch.object(rolebench, "load_tasks", return_value=[task]):
                with patch.object(rolebench, "run_one", return_value=candidate):
                    with patch.object(rolebench, "run_fixed_judge", side_effect=judge):
                        self.assertEqual(rolebench.run_command(args), 0)
            path = root / "run" / "judge_evaluations.jsonl"
            persisted = json.loads(path.read_text(encoding="utf-8"))
            persisted["dimensions"]["correctness"]["score"] = 0
            path.write_text(json.dumps(persisted) + "\n", encoding="utf-8")
            with patch.object(rolebench, "load_tasks", return_value=[task]):
                with self.assertRaisesRegex(rolebench.RoleBenchError, "row content digest does not match"):
                    rolebench.run_command(self.make_run_args(root, pricing_file=str(pricing), judge_model="judge/model", resume=True))

    def test_resume_keeps_run_wide_parallel_latency_context(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            task = {"id": "task-1", "role": "task", "prompt": "do"}
            calls = 0
            lock = threading.Lock()

            def interrupted_candidate(*_args):
                nonlocal calls
                with lock:
                    calls += 1
                    call = calls
                if call == 4:
                    raise KeyboardInterrupt()
                return {"status": "pass", "execution_status": "completed", "provider_call_started": False, "output": f"done-{call}"}

            with patch.object(rolebench, "load_tasks", return_value=[task]):
                with patch.object(rolebench, "run_one", side_effect=interrupted_candidate):
                    self.assertEqual(rolebench.run_command(self.make_run_args(root, runs=4, jobs=3)), 130)
            self.assertEqual(len(rolebench.load_jsonl(root / "run" / "results.jsonl", "results.jsonl")), 3)
            with patch.object(rolebench, "load_tasks", return_value=[task]):
                with patch.object(rolebench, "run_one", return_value={"status": "pass", "execution_status": "completed", "provider_call_started": False, "output": "last"}):
                    self.assertEqual(rolebench.run_command(self.make_run_args(root, runs=4, jobs=3, resume=True)), 0)
            manifest = json.loads((root / "run" / "manifest.json").read_text(encoding="utf-8"))
            summary = json.loads((root / "run" / "summary.json").read_text(encoding="utf-8"))
            report = (root / "run" / "report.md").read_text(encoding="utf-8")
            self.assertEqual(manifest["parallelism"]["effective_jobs"], 3)
            self.assertEqual(manifest["parallelism"]["resume_segment"]["effective_jobs"], 1)
            self.assertEqual(summary["effective_jobs"], 3)
            self.assertEqual(summary["execution_effective_jobs"], 1)
            self.assertIn("Latency mode: `concurrent`", report)
            self.assertIn("up to 3 in-flight calls", report)

    @patch("rolebench.subprocess.run")
    def test_run_one_records_cleanup_failure_without_overwriting_result(self, run):
        run.return_value = SimpleNamespace(returncode=0, stdout=json.dumps({"output": "done"}), stderr="")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            assignment = {"id": "a", "mode": "role-isolated", "role": "task", "candidate": "p/a", "mapping": {"task": "p/a"}}
            task = {"id": "task-1", "role": "task", "prompt": "do"}
            with patch.object(rolebench, "remove_temp_root", side_effect=OSError("busy")):
                result = rolebench.run_one("omp", assignment, task, 1, root / "output", 30, root / "config.yml", "direct", None)
            warning = next(item for item in result["warnings"] if item["type"] == "workspace_cleanup_failed")
            self.assertEqual(result["status"], "pass")
            self.assertEqual(result["output"], "done")
            self.assertIn("omp-rolebench-", warning["path"])
            rolebench.shutil.rmtree(warning["path"], ignore_errors=True)

    def test_run_one_releases_workspace_after_success_timeout_and_interrupt(self):
        assignment = {"id": "a", "mode": "role-isolated", "role": "task", "candidate": "p/a", "mapping": {"task": "p/a"}}
        task = {"id": "task-1", "role": "task", "prompt": "do"}
        cases = [
            SimpleNamespace(returncode=0, stdout=json.dumps({"output": "done"}), stderr=""),
            subprocess.TimeoutExpired(["omp"], 30),
            KeyboardInterrupt(),
        ]
        for index, outcome in enumerate(cases):
            workspace = Path(tempfile.mkdtemp(prefix="omp-rolebench-"))
            try:
                with self.subTest(index=index):
                    with tempfile.TemporaryDirectory() as directory:
                        root = Path(directory)
                        with patch.object(rolebench.tempfile, "mkdtemp", return_value=str(workspace)):
                            with patch.object(rolebench.subprocess, "run", side_effect=outcome if isinstance(outcome, BaseException) else None, return_value=None if isinstance(outcome, BaseException) else outcome):
                                if isinstance(outcome, KeyboardInterrupt):
                                    with self.assertRaises(KeyboardInterrupt):
                                        rolebench.run_one("omp", assignment, task, 1, root / "output", 30, root / "config.yml", "direct", None)
                                else:
                                    rolebench.run_one("omp", assignment, task, 1, root / "output", 30, root / "config.yml", "direct", None)
                        self.assertFalse(workspace.exists())
            finally:
                rolebench.shutil.rmtree(workspace, ignore_errors=True)

    def test_run_one_releases_workspace_when_evidence_capture_is_interrupted(self):
        workspace = Path(tempfile.mkdtemp(prefix="omp-rolebench-"))
        assignment = {"id": "a", "mode": "role-isolated", "role": "task", "candidate": "p/a", "mapping": {"task": "p/a"}}
        task = {"id": "task-1", "role": "task", "prompt": "do"}
        completed = SimpleNamespace(returncode=0, stdout=json.dumps({"output": "done"}), stderr="")
        try:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                with patch.object(rolebench.tempfile, "mkdtemp", return_value=str(workspace)):
                    with patch.object(rolebench.subprocess, "run", return_value=completed):
                        with patch.object(rolebench, "snapshot_workspace", side_effect=[{"files": {}}, KeyboardInterrupt()]):
                            with self.assertRaises(KeyboardInterrupt):
                                rolebench.run_one("omp", assignment, task, 1, root / "output", 30, root / "config.yml", "direct", None)
                self.assertFalse(workspace.exists())
        finally:
            rolebench.shutil.rmtree(workspace, ignore_errors=True)

    def test_cleanup_failure_during_interrupt_preserves_paid_result_for_coordinator(self):
        workspace = Path(tempfile.mkdtemp(prefix="omp-rolebench-"))
        assignment = {"id": "a", "mode": "role-isolated", "role": "task", "candidate": "p/a", "mapping": {"task": "p/a"}}
        task = {"id": "task-1", "role": "task", "prompt": "do"}
        completed = SimpleNamespace(returncode=0, stdout=json.dumps({"output": "done"}), stderr="")
        try:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                with patch.object(rolebench, "verify_harness_fingerprint"):
                    with patch.object(rolebench.tempfile, "mkdtemp", return_value=str(workspace)):
                        with patch.object(rolebench.subprocess, "run", return_value=completed):
                            with patch.object(rolebench, "snapshot_workspace", side_effect=[{"files": {}}, KeyboardInterrupt()]):
                                with patch.object(rolebench, "remove_temp_root", side_effect=OSError("busy")):
                                    outcome = rolebench.execute_candidate_cell(
                                        "omp", assignment, task, 1, root / "output", 30, root / "config.yml",
                                        "direct", None, False, root / "artifacts", {"digest": "same"},
                                    )
            self.assertEqual(outcome["row"]["status"], "pass")
            self.assertTrue(outcome["row"]["provider_call_started"])
            self.assertEqual(outcome["stop_reason"], "workspace_cleanup_failed")
            self.assertEqual(outcome["row"]["warnings"][-1]["path"], str(workspace))
        finally:
            rolebench.shutil.rmtree(workspace, ignore_errors=True)

    def test_cleanup_failure_stops_scheduling_and_reports_unreleased_path(self):
        workspace = Path(tempfile.mkdtemp(prefix="omp-rolebench-"))
        try:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                workspace_identity = rolebench.cleanup_identity_payload(rolebench.temp_root_identity(workspace))
                row = {
                    "status": "pass",
                    "execution_status": "completed",
                    "provider_call_started": False,
                    "output": "done",
                    "warnings": [{"type": "workspace_cleanup_failed", "path": str(workspace), "identity": workspace_identity, "error": "busy"}],
                }
                task = {"id": "task-1", "role": "task", "prompt": "do"}
                with patch.object(rolebench, "load_tasks", return_value=[task]):
                    with patch.object(rolebench, "run_one", return_value=row) as run:
                        with patch.object(rolebench, "remove_temp_root", side_effect=OSError("still busy")):
                            self.assertEqual(rolebench.run_command(self.make_run_args(root, runs=2)), 4)
                summary = json.loads((root / "run" / "summary.json").read_text(encoding="utf-8"))
                self.assertEqual(run.call_count, 1)
                self.assertEqual(summary["stopped_reason"], "workspace_cleanup_failed")
                self.assertEqual(summary["workspace_cleanup"]["unreleased_paths"], [str(workspace)])
                self.assertEqual(summary["workspace_cleanup"]["status"], "failed")
        finally:
            rolebench.shutil.rmtree(workspace, ignore_errors=True)

    def test_candidate_raw_write_failure_keeps_observed_cost_and_skips_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            task = {"id": "task-1", "role": "task", "prompt": "do"}
            evaluation = {"assignment_id": "x", "task_id": "task-1", "replicate": 1}
            row = {
                "status": "pass",
                "execution_status": "completed",
                "provider_call_started": True,
                "output": "done",
                "usage": {"cost": {"total": 0.2}},
                "_evaluation": evaluation,
            }
            original_append = rolebench.append_result

            def append(path, value):
                if path.name == "results.jsonl":
                    raise OSError("disk full")
                original_append(path, value)

            with patch.object(rolebench, "load_tasks", return_value=[task]):
                with patch.object(rolebench, "run_one", return_value=row):
                    with patch.object(rolebench, "append_result", side_effect=append):
                        self.assertEqual(rolebench.run_command(self.make_run_args(root)), 4)
            summary = json.loads((root / "run" / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["actual_cost_usd"], 0.2)
            self.assertEqual(summary["stopped_reason"], "result_write_failed")
            self.assertFalse((root / "run" / "evaluations.jsonl").exists())

    def test_judge_raw_write_failure_does_not_create_orphan_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pricing = root / "pricing.json"
            pricing.write_text(json.dumps({
                "p/a": {"input_per_million": 1, "output_per_million": 1},
                "judge/model": {"input_per_million": 1, "output_per_million": 1},
            }), encoding="utf-8")
            task = {"id": "task-1", "role": "task", "prompt": "do", "dimension_targets": ["correctness"]}
            candidate_evaluation = {"assignment_id": "x", "task_id": "task-1", "replicate": 1, "checks": [], "dimensions": {}}
            candidate = {"status": "pass", "execution_status": "completed", "provider_call_started": False, "output": "done", "_evaluation": candidate_evaluation}
            judge_row = {"status": "pass", "execution_status": "completed", "provider_call_started": True, "usage": {"cost": {"total": 0.3}}}
            judge_evaluation = {"assignment_id": "x", "task_id": "task-1", "replicate": 1, "checks": [], "dimensions": {}}
            original_append = rolebench.append_result

            def append(path, value):
                if path.name == "judge_results.jsonl":
                    raise OSError("disk full")
                original_append(path, value)

            args = self.make_run_args(root, pricing_file=str(pricing), judge_model="judge/model")
            with patch.object(rolebench, "load_tasks", return_value=[task]):
                with patch.object(rolebench, "run_one", return_value=candidate):
                    with patch.object(rolebench, "run_fixed_judge", return_value=(judge_row, judge_evaluation)):
                        with patch.object(rolebench, "append_result", side_effect=append):
                            self.assertEqual(rolebench.run_command(args), 4)
            summary = json.loads((root / "run" / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["actual_cost_usd"], 0.3)
            self.assertEqual(summary["stopped_reason"], "judge_write_failed")
            self.assertFalse((root / "run" / "judge_evaluations.jsonl").exists())

    def test_remove_temp_root_refuses_replaced_symlink(self):
        target = Path(tempfile.mkdtemp(prefix="omp-rolebench-"))
        original = Path(tempfile.mkdtemp(prefix="omp-rolebench-"))
        original_identity = rolebench.temp_root_identity(original)
        marker = target / "keep.txt"
        marker.write_text("keep", encoding="utf-8")
        rolebench.shutil.rmtree(original)
        original.symlink_to(target, target_is_directory=True)
        try:
            with self.assertRaisesRegex(rolebench.RoleBenchError, "replaced temporary path"):
                rolebench.remove_temp_root(original, original_identity)
            self.assertEqual(marker.read_text(encoding="utf-8"), "keep")
        finally:
            original.unlink(missing_ok=True)
            rolebench.shutil.rmtree(target, ignore_errors=True)

    def test_remove_temp_root_refuses_replaced_directory(self):
        original = Path(tempfile.mkdtemp(prefix="omp-rolebench-"))
        moved = original.with_name(original.name + "-moved")
        identity = rolebench.temp_root_identity(original)
        original.rename(moved)
        original.mkdir()
        marker = original / "keep.txt"
        marker.write_text("keep", encoding="utf-8")
        try:
            with self.assertRaisesRegex(rolebench.RoleBenchError, "replaced temporary directory"):
                rolebench.remove_temp_root(original, identity)
            self.assertEqual(marker.read_text(encoding="utf-8"), "keep")
        finally:
            rolebench.shutil.rmtree(original, ignore_errors=True)
            rolebench.shutil.rmtree(moved, ignore_errors=True)

    def test_remove_temp_root_reports_a_moved_workspace_as_unreleased(self):
        original = Path(tempfile.mkdtemp(prefix="omp-rolebench-"))
        moved = original.with_name(original.name + "-moved")
        identity = rolebench.temp_root_identity(original)
        original.rename(moved)
        try:
            with self.assertRaisesRegex(rolebench.RoleBenchError, "disappeared before cleanup"):
                rolebench.remove_temp_root(original, identity)
            self.assertTrue(moved.is_dir())
        finally:
            rolebench.shutil.rmtree(moved, ignore_errors=True)

    def test_workspace_size_follows_directory_symlinks_like_copytree(self):
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as external:
            root = Path(directory)
            external_root = Path(external)
            payload = external_root / "payload.bin"
            payload.write_bytes(b"x" * (128 * 1024))
            (root / "linked").symlink_to(external_root, target_is_directory=True)
            self.assertEqual(rolebench.copied_tree_size(root, root / "output"), 128 * 1024)

    def test_provider_parallelism_accounts_for_candidate_judge_dependency(self):
        self.assertEqual(
            rolebench.provider_parallelism_plan(2, candidate_calls=1, ready_judge_calls=0, judge_enabled=True),
            {"effective_jobs": 1, "provider_calls": 2, "estimated_provider_waves": 2},
        )
        self.assertEqual(
            rolebench.provider_parallelism_plan(2, candidate_calls=1, ready_judge_calls=1, judge_enabled=True),
            {"effective_jobs": 2, "provider_calls": 3, "estimated_provider_waves": 2},
        )
        self.assertEqual(
            rolebench.provider_parallelism_plan(2, candidate_calls=3, ready_judge_calls=0, judge_enabled=True),
            {"effective_jobs": 2, "provider_calls": 6, "estimated_provider_waves": 3},
        )

    def test_resume_without_judge_rejects_residual_judge_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            task = {"id": "task-1", "role": "task", "prompt": "do"}
            with patch.object(rolebench, "load_tasks", return_value=[task]):
                self.assertEqual(rolebench.run_command(self.make_run_args(root, execute=False, confirm=False)), 0)
            manifest = json.loads((root / "run" / "manifest.json").read_text(encoding="utf-8"))
            assignment_id = manifest["assignments"][0]["id"]
            candidate = {
                "assignment_id": assignment_id,
                "task_id": "task-1",
                "replicate": 1,
                "provider_call_started": False,
                "status": "pass",
                "execution_status": "completed",
            }
            judge = {**candidate, "model": "unexpected/judge", "evaluator": "fixed_judge"}
            rolebench.append_result(root / "run" / "results.jsonl", candidate)
            rolebench.append_result(root / "run" / "judge_results.jsonl", judge)
            with patch.object(rolebench, "load_tasks", return_value=[task]):
                with self.assertRaisesRegex(rolebench.RoleBenchError, "audit rows do not belong"):
                    rolebench.run_command(self.make_run_args(root, resume=True))

    def test_coordinator_preserves_completed_rows_when_persistence_is_interrupted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            task = {"id": "task-1", "role": "task", "prompt": "do"}
            row = {
                "status": "pass",
                "execution_status": "completed",
                "provider_call_started": True,
                "output": "done",
                "usage": {"cost": {"total": 0.2}},
            }
            original_append = rolebench.append_result
            interrupted = False

            def append(path, value):
                nonlocal interrupted
                if path.name == "results.jsonl" and not interrupted:
                    interrupted = True
                    raise KeyboardInterrupt()
                original_append(path, value)

            with patch.object(rolebench, "load_tasks", return_value=[task]):
                with patch.object(rolebench, "run_one", return_value=row):
                    with patch.object(rolebench, "append_result", side_effect=append):
                        self.assertEqual(rolebench.run_command(self.make_run_args(root, runs=2, jobs=2)), 130)
            results = rolebench.load_jsonl(root / "run" / "results.jsonl", "results.jsonl")
            summary = json.loads((root / "run" / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(len(results), 2)
            self.assertEqual(summary["actual_cost_usd"], 0.4)
            self.assertEqual(summary["stopped_reason"], "interrupted")

    def test_submit_interrupt_drains_and_persists_registered_future(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            task = {"id": "task-1", "role": "task", "prompt": "do"}
            row = {
                "status": "pass",
                "execution_status": "completed",
                "provider_call_started": True,
                "output": "done",
                "usage": {"cost": {"total": 0.2}},
            }
            original_submit = rolebench.ThreadPoolExecutor.submit
            submit_count = 0

            def submit(pool, *args, **kwargs):
                nonlocal submit_count
                submit_count += 1
                if submit_count == 2:
                    raise KeyboardInterrupt()
                return original_submit(pool, *args, **kwargs)

            with patch.object(rolebench, "load_tasks", return_value=[task]):
                with patch.object(rolebench, "run_one", return_value=row):
                    with patch.object(rolebench.ThreadPoolExecutor, "submit", new=submit):
                        self.assertEqual(rolebench.run_command(self.make_run_args(root, runs=2, jobs=2)), 130)
            results = rolebench.load_jsonl(root / "run" / "results.jsonl", "results.jsonl")
            summary = json.loads((root / "run" / "summary.json").read_text(encoding="utf-8"))
            self.assertEqual(len(results), 1)
            self.assertEqual(summary["actual_cost_usd"], 0.2)
            self.assertEqual(summary["stopped_reason"], "interrupted")

    def test_cleanup_retry_interrupt_still_writes_summary(self):
        workspace = Path(tempfile.mkdtemp(prefix="omp-rolebench-"))
        try:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                task = {"id": "task-1", "role": "task", "prompt": "do"}
                workspace_identity = rolebench.cleanup_identity_payload(rolebench.temp_root_identity(workspace))
                row = {
                    "status": "pass",
                    "execution_status": "completed",
                    "provider_call_started": False,
                    "output": "done",
                    "warnings": [{"type": "workspace_cleanup_failed", "path": str(workspace), "identity": workspace_identity, "error": "busy"}],
                }
                with patch.object(rolebench, "load_tasks", return_value=[task]):
                    with patch.object(rolebench, "run_one", return_value=row):
                        with patch.object(rolebench, "remove_temp_root", side_effect=KeyboardInterrupt()):
                            self.assertEqual(rolebench.run_command(self.make_run_args(root)), 130)
                summary = json.loads((root / "run" / "summary.json").read_text(encoding="utf-8"))
                self.assertEqual(summary["stopped_reason"], "interrupted")
                self.assertEqual(summary["workspace_cleanup"]["unreleased_paths"], [str(workspace)])
        finally:
            rolebench.shutil.rmtree(workspace, ignore_errors=True)

    def test_judge_internal_error_preserves_started_call_and_cost(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pricing = root / "pricing.json"
            pricing.write_text(json.dumps({
                "p/a": {"input_per_million": 1, "output_per_million": 1},
                "judge/model": {"input_per_million": 1, "output_per_million": 1},
            }), encoding="utf-8")
            task = {"id": "task-1", "role": "task", "prompt": "do", "dimension_targets": ["correctness"]}
            candidate = {"status": "pass", "execution_status": "completed", "provider_call_started": False, "output": "done"}
            judge_payload = {
                "dimensions": {"correctness": {"status": "pass", "score": 2, "evidence": "ok"}},
                "usage": {"cost": {"total": 0.4}},
            }
            completed = SimpleNamespace(returncode=0, stdout=json.dumps(judge_payload), stderr="")
            args = self.make_run_args(root, pricing_file=str(pricing), judge_model="judge/model", max_cost_usd=1)
            with patch.object(rolebench, "load_tasks", return_value=[task]):
                with patch.object(rolebench, "run_one", return_value=candidate):
                    with patch.object(rolebench.subprocess, "run", return_value=completed):
                        with patch.object(rolebench, "build_judge_evaluation", side_effect=RuntimeError("evaluation broke")):
                            self.assertEqual(rolebench.run_command(args), 4)
            judge_rows = rolebench.load_jsonl(root / "run" / "judge_results.jsonl", "judge_results.jsonl")
            summary = json.loads((root / "run" / "summary.json").read_text(encoding="utf-8"))
            self.assertTrue(judge_rows[0]["provider_call_started"])
            self.assertEqual(judge_rows[0]["usage"]["cost"]["total"], 0.4)
            self.assertEqual(judge_rows[0]["error_type"], "judge_internal")
            self.assertEqual(summary["actual_cost_usd"], 0.4)
            self.assertEqual(summary["stopped_reason"], "harness_judge_failed")

    def test_judge_cleanup_failure_preserves_response_usage_and_path(self):
        task = {"id": "task-1", "role": "task", "prompt": "do", "dimension_targets": ["correctness"]}
        result = {
            "assignment_id": "a",
            "task_id": "task-1",
            "role": "task",
            "model": "p/a",
            "replicate": 1,
            "execution_status": "completed",
            "output": "done",
        }
        deterministic = {"checks": [], "dimensions": {}}
        payload = {
            "dimensions": {"correctness": {"status": "pass", "score": 2, "evidence": "ok"}},
            "usage": {"cost": {"total": 0.4}},
        }
        completed = SimpleNamespace(returncode=0, stdout=json.dumps(payload), stderr="")
        warning_path = None
        try:
            with patch.object(rolebench.subprocess, "run", return_value=completed):
                with patch.object(rolebench, "remove_temp_root", side_effect=OSError("busy")):
                    raw, evaluation = rolebench.run_fixed_judge("omp", "judge/model", task, result, deterministic)
            warning = raw["warnings"][0]
            warning_path = warning["path"]
            self.assertEqual(raw["stdout"], json.dumps(payload))
            self.assertEqual(raw["usage"]["cost"]["total"], 0.4)
            self.assertTrue(raw["provider_call_started"])
            self.assertEqual(warning["type"], "workspace_cleanup_failed")
            self.assertEqual(evaluation["evaluation_status"], "completed")
        finally:
            if warning_path:
                rolebench.shutil.rmtree(warning_path, ignore_errors=True)

    def test_judge_cleanup_failure_is_retried_after_drain(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pricing = root / "pricing.json"
            pricing.write_text(json.dumps({
                "p/a": {"input_per_million": 1, "output_per_million": 1},
                "judge/model": {"input_per_million": 1, "output_per_million": 1},
            }), encoding="utf-8")
            task = {"id": "task-1", "role": "task", "prompt": "do", "dimension_targets": ["correctness"]}
            candidate = {"status": "pass", "execution_status": "completed", "provider_call_started": False, "output": "done"}
            payload = {
                "dimensions": {"correctness": {"status": "pass", "score": 2, "evidence": "ok"}},
                "usage": {"cost": {"total": 0.4}},
            }
            completed = SimpleNamespace(returncode=0, stdout=json.dumps(payload), stderr="")
            original_remove = rolebench.remove_temp_root
            failed_path = None

            def remove(path, expected_identity):
                nonlocal failed_path
                if "omp-rolebench-judge-" in path.name and failed_path is None:
                    failed_path = path
                    raise OSError("busy")
                original_remove(path, expected_identity)

            args = self.make_run_args(root, pricing_file=str(pricing), judge_model="judge/model")
            with patch.object(rolebench, "load_tasks", return_value=[task]):
                with patch.object(rolebench, "run_one", return_value=candidate):
                    with patch.object(rolebench.subprocess, "run", return_value=completed):
                        with patch.object(rolebench, "remove_temp_root", side_effect=remove):
                            self.assertEqual(rolebench.run_command(args), 4)
            summary = json.loads((root / "run" / "summary.json").read_text(encoding="utf-8"))
            judge_rows = rolebench.load_jsonl(root / "run" / "judge_results.jsonl", "judge_results.jsonl")
            self.assertEqual(judge_rows[0]["usage"]["cost"]["total"], 0.4)
            self.assertEqual(summary["workspace_cleanup"]["status"], "recovered-after-failure")
            self.assertEqual(summary["workspace_cleanup"]["unreleased_paths"], [])
            self.assertIsNotNone(failed_path)
            self.assertFalse(failed_path.exists())

    def test_reports_expose_workspace_and_budget_boundaries(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            task = {"id": "task-1", "role": "task", "prompt": "do"}
            row = {"status": "pass", "execution_status": "completed", "provider_call_started": False, "output": "done"}
            with patch.object(rolebench, "load_tasks", return_value=[task]):
                with patch.object(rolebench, "run_one", return_value=row):
                    self.assertEqual(rolebench.run_command(self.make_run_args(root, runs=2, jobs=2, max_cost_usd=1)), 2)
            report = (root / "run" / "report.md").read_text(encoding="utf-8")
            payload = json.loads((root / "run" / "report.json").read_text(encoding="utf-8"))
            self.assertIn("Workspace estimate", report)
            self.assertIn("Budget: cap $1.000000", report)
            self.assertIn("up to 2 in-flight calls", report)
            self.assertEqual(payload["execution_context"]["max_cost_usd"], 1)
            self.assertIn("workspace_estimated_peak_bytes", payload["execution_context"])


if __name__ == "__main__":
    unittest.main()
