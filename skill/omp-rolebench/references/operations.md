# Execution, Recovery, and Compatibility

This reference defines the operational boundary of a RoleBench run. Evaluation semantics remain in `evaluation-v1.md` and `evaluation-v2.md`.

## Failure classification

| Failure | Recorded as | Run behavior |
|---|---|---|
| Candidate OMP exit, timeout, empty output, or delegated-task failure | Candidate execution error | Continue to later cells when uncapped; a capped run stops if provider cost is unavailable |
| OMP event JSON cannot be parsed, executor setup/internal failure, or missing required fixture | Harness/integration failure | Preserve the raw row and stop the complete run |
| Fixed-judge exit, timeout, or malformed answer | Judge `unknown` | Continue unless the cost cap becomes unverifiable or reached |
| Snapshot, artifact, deterministic evaluator, JSONL write, or unexpected harness failure | Harness failure | Preserve completed evidence and stop the complete run |
| Temporary cell workspace cannot be removed | Harness cleanup failure | Preserve the result, record the exact path, stop new calls, drain in-flight calls, retry cleanup, and report unresolved paths |
| Harness/skill fingerprint changes after execution begins | Harness integrity failure | Stop immediately; do not combine later cells with the run |
| Process interruption or lost terminal connection | Missing cells | Persisted JSONL rows remain recoverable with `--resume` |

Never repair the harness and continue the same matrix in place. A harness failure requires a new output directory and a full rerun after the user authorizes maintenance. `--resume` is only for an interruption with the same experiment and unchanged harness.

The CLI exits `4` for harness, evidence, or audit-write failures, and `130` for an interrupted process. Candidate/provider failures remain benchmark observations: an uncapped run continues and exits `0` after completing the requested matrix, while budget gates retain their documented stopped or blocked status.

## Cell-level resume

Repeat the original command with the same options, add `--resume`, and keep `--execute --confirm`:

```bash
python3 omp-rolebench/scripts/rolebench.py run \
  <same selection, tasks, executor, judge, pricing, cap, run count, and jobs> \
  --execute --confirm --resume \
  --output runs/interrupted-run
```

The manifest contains an experiment signature over the normalized tasks, assignments, models, executor/controller/judge settings, pricing estimate, timeouts, seed, jobs, source workspace/image content fingerprints (computed with the same dereferenced symlink and nested-output exclusions as workspace copying), and evaluator configuration. It also contains a fingerprint of `SKILL.md`, `scripts/`, `references/`, `assets/`, and `agents/`. Resume fails before a provider call if either value differs. Source fixture changes are not a compatible resume: a new output directory and full matrix are required.

`results.jsonl` is the candidate-call commit record. A persisted result cell is not called again. If its deterministic evaluation was not persisted before interruption, RoleBench writes an explicit `unknown` evaluation because the deleted temporary workspace cannot be reconstructed. Existing judge results are normalized locally when only their evaluation row is missing; a judge is called only when its raw row is absent. Duplicate or orphaned audit rows fail loudly.

The resumed plan shows completed and remaining candidate/judge cells and the effective width/waves of the remaining execution segment. Run-wide effective concurrency and latency classification stay fixed to the original full matrix, while observed costs from persisted rows are carried forward before any new call. All four audit streams are checked before the manifest is rewritten or another provider call starts: candidate and judge raw rows carry independent content digests, evaluation rows carry independent content plus raw-result digests, candidate and judge evaluations must match their cell/model metadata, and fixed-judge evaluations must reproduce from the exact accepted payload retained in the raw judge row.

JSONL persistence is at-least-once at the interruption boundary, not exactly-once. If a process or machine dies after a provider accepted a request but before its result row was flushed, that cell has no auditable completion record and `--resume` will issue it again after confirmation. Treat the resumed plan's remaining calls as potential additional cost and inspect provider-side usage when this ambiguity matters.

## Parallel direct cells

Each candidate cell is identified by `(assignment_id, task_id, replicate)`. `--jobs N` applies a single concurrency ceiling to direct candidate calls and their downstream fixed-judge calls; a judge cannot enter the queue until the candidate raw row and deterministic evaluation are persisted. `--jobs 1` preserves the original candidate-then-judge order for each cell. With `jobs > 1`, remaining candidates take priority over newly ready judges so the dependency-aware provider-wave estimate matches the queue policy. Effective width is based on calls that are actually runnable instead of total calls. The coordinator is the only JSONL writer and budget owner. Parallel mode uses a seed-stable queue and allows completion order to differ. Delegated execution rejects `jobs > 1`.

When a budget or harness stop is observed, RoleBench submits no new provider calls and waits for already-running calls to finish so their evidence, cost, and workspaces can be handled. With concurrency, those in-flight calls can make actual cost exceed the cap. Reported latency is concurrency/load latency; do not compare it directly with a serial run.

Every candidate provider process has its own temporary config and copied workspace. Every fixed judge has its own minimal temporary config and empty temporary cwd, with extensions, skills, rules, and tools disabled. A workspace stays present until its process exits and all applicable evidence collection finishes, then its complete temporary root is removed on every success, error, timeout, or interruption path. Removal is limited to the lexical harness-created `omp-rolebench-*` temporary root and requires its device/inode identity to match the directory created for that cell, so a path replaced by a symlink, non-directory, or different directory is never removed. A missing original path is also a cleanup failure because the same inode may have been moved and still contain the copied workspace. If cleanup fails, the raw result remains valid, the row records `workspace_cleanup_failed`, its path, and its creation identity; new calls stop, existing calls drain, and the coordinator retries the same identity after draining. `summary.json.workspace_cleanup.unreleased_paths` is the final manual-cleanup list. Plan-time disk estimates follow directory symlinks because workspace copying does, reject symlink cycles, and show the largest copied cell and approximate peak across effective jobs; real peak can differ when tasks create files.

## Harness integrity boundary

Every candidate cell runs in a temporary copied workspace and temporary OMP config. The skill repository is not the task working directory or a fixture exposed in the prompt. RoleBench checks its recorded skill-tree fingerprint immediately before and after each new candidate and judge call. Snapshot, artifact, and evaluator warnings stop the run rather than being attributed to the model.

This is change detection, not host isolation. OMP tools and explicitly approved command invariants still run as local processes and may access host paths permitted by the operating environment. Use an external sandbox for hostile prompts, commands, network, credentials, or filesystem side effects.

## OMP compatibility

The harness requires OMP print mode with `-p`, `--mode json`, `--no-session`, `--no-title`, `--model`, `--config`, `--append-system-prompt`, `--cwd`, `--auto-approve`, `--thinking`, fixed-judge `--no-tools`, `--no-extensions`, `--no-skills`, `--no-rules`, and delegated `--tools task`. Model discovery requires `omp models --json`.

OMP event extraction selects the final text segment from the last completed assistant turn, including finalized `message_update` text when `message_end` contains merged earlier chatter. It aggregates usage only when every completed assistant message contains a usage object. Missing usage stays unknown. Before a paid benchmark after an OMP upgrade:

1. Run `omp --version` and inspect `omp -p --help` for the required flags.
2. Run `rolebench.py discover --omp-bin omp` and confirm the catalog normalizes to provider/model selectors.
3. Execute the unit suite and a `tests/fake_omp` smoke run to validate argv, event extraction, artifacts, reports, resume, and harness integrity without provider calls.
4. Preview the real matrix again and obtain consent for the current plan and cost before any provider call.

OMP 18.0.11 was checked against this contract on 2026-09-01. This is a CLI/event-shape compatibility statement, not a claim that every provider reports cost or usage.
