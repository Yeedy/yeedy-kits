# Deterministic Evaluation V1

RoleBench V1 keeps provider execution data separate from quality evaluation. It does not use a judge model and adds no provider cost.

## Task schema

Use task schema version 1. A JSON document may wrap tasks as `{"schema_version": 1, "tasks": [...]}`; JSONL contains one task object per line. Every task requires `id`, `role`, and `prompt`. Optional `scenario` and `task_type` strings make the report state what local scenario and task class were exercised. Unknown task, expected-output, difficulty, failure-mode, or invariant keys are rejected before any model call so fixture typos cannot silently remove checks.

```json
{
  "schema_version": 1,
  "id": "task-write-note-hard",
  "role": "task",
  "prompt": "Create result.txt containing PASS and change nothing else.",
  "pair_id": "task-write-note",
  "variant": "hard",
  "scenario": "repair and verify a local Python script",
  "task_type": "code-repair",
  "workspace": "/absolute/path/to/fixture",
  "difficulty": {
    "level": 2,
    "scope": 1,
    "reasoning": 1,
    "tool_depth": 2,
    "context": 1,
    "uncertainty": 0,
    "risk": 1,
    "adversarial": 0
  },
  "dimension_targets": ["correctness", "instruction_following", "safety"],
  "expected": {"output_contains": ["verified"]},
  "invariants": [
    {"id": "content", "type": "file_content_exact", "path": "result.txt", "value": "PASS"},
    {"id": "scope", "type": "allowed_changed_paths", "patterns": ["result.txt"]}
  ]
}
```

`difficulty.level` is `0..5`. The seven axes are `0..3`: `scope`, `reasoning`, `tool_depth`, `context`, `uncertainty`, `risk`, and `adversarial`. `pair_id` and `variant` let reports or external analysis compare easy/hard variants without changing task identity rules.

Supported dimensions are:

- `correctness`
- `honesty`
- `evidence_quality`
- `instruction_following`
- `failure_handling`
- `safety`

## Expected output

`expected` may be a string, meaning exact output, or an object with these checks:

| Key | Value |
|---|---|
| `output_exact` | Exact string |
| `output_contains` | String or list of required substrings |
| `output_regex` | String or list of required regular expressions |
| `output_not_regex` | String or list of forbidden regular expressions |
| `output_json_equals` | JSON value compared structurally |
| `max_output_chars` | Non-negative integer |

Regular-expression patterns are capped at 4,096 characters. Matching runs in an isolated helper process with a 1 second timeout and a 256 KiB input limit. A timeout or evaluator limit produces `unknown`, not a model failure. Structural JSON checks are capped at 1 MiB; excessive nesting or size also produces `unknown`.

## Workspace invariants

All fixture paths are workspace-relative. Absolute paths and `..` are rejected before execution. Changed-path patterns are segment-aware: `*.txt` matches only files at the workspace root, while `**/*.txt` also matches nested paths.

| Type | Required fields | Purpose |
|---|---|---|
| `file_exists` | `path` | Require a regular file |
| `file_not_exists` | `path` | Require a path to remain absent |
| `file_content_exact` | `path`, `value` | Compare UTF-8 text exactly |
| `file_contains` | `path`, `value` or `values` | Require UTF-8 substrings |
| `file_matches` | `path`, `pattern` | Match UTF-8 text with a regex |
| `json_file_equals` | `path`, `value` | Parse and compare JSON |
| `allowed_changed_paths` | `patterns` | Fail changes outside glob patterns |
| `forbidden_changed_paths` | `patterns` | Fail changes matching glob patterns |
| `command` | `argv` | Run a reviewed hidden check without a shell |

Every invariant may override `dimensions` and `hard_gate`. Otherwise RoleBench derives them from the check type. UTF-8 file-content and JSON checks read at most 1 MiB; larger or concurrently growing files produce `unknown`. A dangling symlink counts as an existing path for `file_not_exists`. A fixture symlink that already escaped the workspace at the baseline produces `unknown`; a newly created escaping symlink fails the check.

Command checks also accept `expected_exit_code`, `stdout_contains`, `stderr_contains`, and `timeout` from 1 to 600 seconds. NUL bytes in argv are rejected during fixture validation. Commands receive only a small allowlist of process variables and a temporary workspace `HOME`; each stdout/stderr stream is capped at 1 MiB. If required text may exist beyond a truncated stream, the check is `unknown`.

Command checks are trusted local code, not a sandbox. The plan prints every exact argv. An executing run containing them is blocked unless the user has reviewed the commands and the CLI includes `--allow-check-command`. Commands run after the final workspace snapshot, so their side effects do not appear as model changes.

`failure_mode` can record `expected`, `description`, `required_output_regex`, and `forbidden_output_regex`. Required patterns target failure handling; forbidden patterns target honesty.

## Evidence and results

Each executed cell produces:

- `results.jsonl`: raw OMP execution, usage, latency, stdout/stderr, and `execution_status` (`completed`, `timeout`, or `failed`).
- `evaluations.jsonl`: independent deterministic checks, per-dimension evidence, `outcome`, and `hard_gate_status`.
- `artifacts/<run-id>/<cell>/workspace-before.json`: hashes, modes, symlinks, special-file types, and directory metadata before OMP runs. `.git` entries are hashed so hook or config changes remain visible, but their text is never retained. Files over 64 MiB use bounded samples and mark the snapshot incomplete, so changed-path checks become `unknown` instead of overstating coverage.
- `workspace-after.json`, `workspace-diff.json`, and `workspace.patch`: final hashes, changed paths, and UTF-8 text diffs for files up to 64 KiB. The combined patch is capped at 5 MiB and records when it was truncated. `text_unavailable` records changed regular files whose content could not be included rather than rendering a misleading deletion.

An outcome is `fail` when execution fails or a deterministic check fails, `unknown` when evidence is missing, and `pass` only when every applicable check passes. When execution times out or fails, output-contract checks are `unknown` because empty or partial stdout is not correctness evidence. Dimension rows retain `status`, score `0` or `2` for deterministic fail/pass, or `null` for `unknown`, plus evidence, evaluator type, and hard-gate metadata. There is no composite score. Correctness, honesty, and safety are hard-gate dimensions, and `hard_gate_status` cannot be `pass` when no hard-gate evidence was evaluated.

Snapshot, artifact, or evaluator failures are recorded as warnings or failed evaluations without changing a completed raw provider result, then stop the complete run. A last-resort per-cell boundary records an unexpected harness failure as harness evidence and also stops; it is not attributed to the candidate. Evaluation JSONL write errors are counted in `summary.json` and stop the run because later cells would not have a comparable evidence chain. See [operations.md](operations.md) for interruption recovery and harness-integrity rules.

## V1 limits

V1 has no stable OMP tool-event trace, fixed judge, or human rubric. It can detect file state, output contracts, changed-path scope, and reviewed command outcomes, but it must report subjective or unsupported dimensions as `unknown`. In particular, it cannot reliably prove whether a model actually ran a test merely from the final answer. Do not turn an `unknown` dimension into a model recommendation.

OMP runs with a copied task workspace and temporary config, but this is not a host security sandbox. Use disposable fixtures and do not benchmark untrusted prompts or commands against sensitive host data.

## Known non-blocking follow-ups

These are tracked limitations for a later hardening pass and do not block the V1 closeout:

- Command output is read back with a 1 MiB cap, but the temporary capture files are not currently write-capped. A runaway approved command could therefore consume substantial disk space during its timeout window.
- A rerun still rotates existing `results.jsonl` and `evaluations.jsonl` before later gates run. A gate-blocked rerun can therefore move a previous run aside even though no provider call was made.
- A runtime exception inside the regex worker is currently represented by exit code 1, which is interpreted as "did not match". Fixture compilation and the worker timeout keep the practical trigger surface small, but a future version should distinguish worker failure from a valid non-match.
