---
name: omp-rolebench
description: Benchmark Oh My Pi model-role assignments. Use when comparing all models, models from selected providers, or explicitly named models across all or selected OMP roles, generating isolated/all-same assignment matrices, running repeatable role tasks, producing a cost/quality report, measuring raw provider response performance (TTFT, throughput, prompt cache) with `omp bench`, or measuring instruction following/working memory with `omp if-bench`.
---

# OMP RoleBench

## Overview

Use `scripts/rolebench.py` to discover the current OMP model catalog, resolve `modelRoles` from an OMP config, generate bounded assignment experiments, optionally execute tasks through OMP, and write raw results, deterministic evaluations, evidence artifacts, optional fixed-judge evaluations, and a Markdown report. The harness never edits the user's OMP config; execution is opt-in because it can call metered providers.

## Quick start

Preview the candidate matrix without making model calls:

```bash
python3 omp-rolebench/scripts/rolebench.py run \
  --models opencode-go/glm-5.3-flash,opencode-go/deepseek-v4-flash-vision-exp \
  --roles smol,task,vision \
  --mode role-isolated \
  --output runs/glm-vs-ds
```

Generate the task suite from recent local work instead of the generic prompts:

```bash
python3 omp-rolebench/scripts/rolebench.py run \
  --models opencode-go/glm-5.3-flash,opencode-go/deepseek-v4-flash-vision-exp \
  --roles smol,task,advisor \
  --task-source recent \
  --recent-workspace . \
  --recent-days 14 \
  --output runs/recent-work
```

Run the same matrix with the OMP subprocess executor only after reviewing it:

```bash
python3 omp-rolebench/scripts/rolebench.py run \
  --providers opencode-go \
  --roles all \
  --mode role-isolated \
  --suite core \
  --runs 3 \
  --jobs 3 \
  --pricing-file pricing.json \
  --max-cost-usd 0.50 \
  --execute \
  --confirm \
  --output runs/20260828-120000
```

Add the V2 fixed judge and gated role recommendations only after reviewing the combined estimate. If `--judge-model` is omitted, recommendations use the model configured for `modelRoles.default` as the judge:

```bash
python3 omp-rolebench/scripts/rolebench.py run \
  --models opencode-go/glm-5.3-flash,opencode-go/deepseek-v4-flash-vision-exp \
  --roles smol,task,advisor \
  --mode role-isolated \
  --suite core \
  --runs 3 \
  --min-recommendation-runs 3 \
  --pricing-file pricing.json \
  --max-cost-usd 0.50 \
  --execute --confirm --recommend \
  --output runs/v2-20260829
```

## Consent and budget gate

Before any provider call, show the user a short plan containing the candidate models, roles, assignment mode, executor, task count, repeat count, independent candidate-cell count, requested/effective `--jobs`, estimated provider waves, maximum copied-workspace size, estimated peak temporary-disk use, request count, estimated input/output tokens, estimated USD cost, deterministic evaluation dimensions, and any exact command-check argv. For delegated execution, also show the controller model and whether it came from `modelRoles.default` or an explicit `--controller-model` override. When a fixed judge is enabled, include the judge model, whether it came from `modelRoles.default` or an explicit `--judge-model` override, rubric version, request count, token estimate, and cost; the displayed total and `--max-cost-usd` gate cover candidate, controller, and judge calls. Ask for explicit approval in the conversation. Before executing, the user can change `--jobs`, controller, or judge model; rerun the preview and obtain consent for that revised plan. If the user states a budget in natural language (for example, "benchmark cost must stay below $0.50"), convert it to `--max-cost-usd 0.50` and repeat the cap in the confirmation request.

The command also enforces the gate: `--execute` without `--confirm` writes the manifest and estimate but exits without making requests. Only pass `--confirm` after the user approves that exact plan. Task fixtures with command checks require a separate `--allow-check-command` flag after the displayed argv has been reviewed; the commands are trusted local code, not a sandbox. A fixed judge always requires a known judge price before execution, even without a cap. `--recommend` without `--judge-model` resolves the judge from `modelRoles.default` and fails early if that role is not configured. A supplied `--max-cost-usd` additionally requires known prices for every candidate, the delegated controller when enabled, and the judge; an unknown estimate, or an estimate above the cap, blocks execution. Pricing files use USD per million tokens:

```json
{
  "opencode-go/glm-5.3-flash": {
    "input_per_million": 0.075,
    "output_per_million": 0.25
  }
}
```

The estimate is a preflight budget, not a promise about tool-heavy context growth. During execution, provider-reported costs are accumulated and later requests stop when the cap is reached. If an attempted provider call returns no cost, including a timeout, a capped run stops after that call with `budget_status: unverifiable`. With `--jobs N`, up to N calls may already be in flight when the stop is observed, so all of them are drained and persisted and their combined cost may exceed the cap. State this exposure before consent.

## Candidate and role selection

Candidate sources are mutually exclusive in intent:

- `--models` accepts explicit comma-separated `provider/model` selectors and skips catalog discovery.
- `--providers all` (default) discovers `omp models --json`; a comma-separated provider list filters the catalog.
- `--catalog-file` supplies a saved `omp models --json` response for offline or reproducible runs.
- `--roles all` reads the `modelRoles` keys from `--config-file` (default `~/.omp/agent/config.yml`). A comma-separated list selects roles explicitly.

Model selectors may include a thinking suffix such as `provider/model:high`; the suffix is recorded separately from model identity. Unknown or malformed explicit selectors fail early.

Thinking variants of the same model are separate benchmark candidates. In `role-isolated` mode, the configured `modelRoles` file is required so non-target selected roles keep their baseline assignment; with an explicit role list, `all-same` and `explicit` modes can run with an assignment file and no baseline config.

## Assignment modes

- `role-isolated`: one candidate is substituted for one selected role while the remaining roles stay at the configured baseline. This is the primary role-fit experiment.
- `all-same`: every selected role uses one candidate at a time. This measures the cost of keeping one model everywhere.
- `explicit`: `--assignment-file` supplies `role: provider/model` entries and is validated against the selected roles.

The harness does not silently enumerate `M^R` combinations. Use isolated results to select a small set of candidate assignments, then validate those with `explicit`.

For task-subagent experiments, see [observed-task-role-ab.md](references/observed-task-role-ab.md) for the known probe pattern and provider-thinking failure case.

## Tasks and execution

`--tasks-file` accepts task schema V1 as JSON or JSONL. Tasks require `id`, `role`, and `prompt`; they may declare `scenario`, `task_type`, difficulty metadata, easy/hard pairing, expected output, workspace invariants, failure modes, and target dimensions. See [evaluation-v1.md](references/evaluation-v1.md) for the complete schema and evaluator behavior. When a tasks file is omitted, choose between `--task-source core` and `--task-source recent`; the CLI default remains `core` for backward compatibility. At the skill interaction layer, if the user has not chosen a source, ask whether to use the generic suite or recent local work before generating the plan.

At every interaction-layer decision point — task source, whether to follow an all-`unknown` run with deterministic fixtures or the fixed judge, judge model choice — present the options with the `ask` tool (recommended option marked) instead of ending the turn with an open text question. Do not stop mid-flow while the next step is a bounded choice the user can pick.

Relative `workspace` paths in a tasks file resolve from that file's directory. The bundled `assets/task-high/tasks-task-high.jsonl` suite provides three reusable `task` fixtures with deterministic output, file, scope, and command checks; command execution still requires explicit `--allow-check-command` approval.

The generic core suite remains a low-cost smoke suite. Its deterministic protocol and workspace tasks provide usable V1 checks, while subjective roles remain `unknown` where evidence is insufficient. For role-allocation decisions, provide paired easy/hard fixtures plus controlled failure or adversarial cases rather than treating core-suite latency as quality.

Use `--task-source recent` to generate one task per selected role and workspace from recent Git commits and working-tree changes. Repeat `--recent-workspace` for multiple workspaces, or pass comma-separated paths. Set `--recent-days` and `--recent-commits` to control the lookback window. Git paths are scoped and normalized relative to each selected workspace; tracked deletions remain context only, and only existing image files become vision fixtures. For a non-Git workspace, the generator falls back to recently modified files. If Git is present but a command fails, the generator records the error and shows a warning in the plan instead of hiding it. It records commit subjects and file paths in the generated prompts and manifest, but does not read OMP sessions, Codex memory, or past chat history. Each generated task keeps its workspace path so every benchmark cell runs against an isolated copy; review `manifest.json` before consenting to execution.

If `--tasks-file` is provided, it takes precedence over `--task-source`.

`--execute` invokes `omp -p --mode json --no-session` and appends each raw result immediately to `results.jsonl`. For OMP event streams, the harness records the final text segment from the last completed assistant turn as model output, including providers that expose that segment through `message_update` before `message_end`; user prompts, thinking, earlier tool chatter, and tool results are not treated as the answer. Usage is summed from every completed assistant message in the cell. It then writes the independent quality result to `evaluations.jsonl`. Each cell gets a temporary config overlay and workspace; the user's normal config is never changed. Before/after hashes, changed paths, and small UTF-8 text patches are retained under `artifacts/<run-id>/` before that cell's workspace is released. Cleanup runs after success, provider failure, timeout, interruption, and evaluator failure; a cleanup failure preserves the candidate evidence, records the exact path as `workspace_cleanup_failed`, stops new scheduling, drains current calls, retries cleanup, and lists any unresolved paths in `summary.json`. Reusing an output directory rotates existing result, evaluation, and report files before writing the new manifest. Every `run` invocation also renders `report.md` and `report.json` in the run directory after persisting `summary.json`; `summary.json` records both report paths and their generation status. Without `--execute`, `manifest.json`, the combined cost estimate, and a scope report are still produced so the experiment can be reviewed before making provider calls.

An independent candidate cell is `(assignment_id, task_id, replicate)`. `--jobs N` sets the maximum number of simultaneous direct candidate or fixed-judge provider calls; the effective width is `min(N, remaining work)`, not a fixed choice such as 2 or 3. The default is `--jobs 1`, which preserves serial behavior. Candidate and judge work share the same global limit, and a judge is queued only after its candidate result and deterministic evaluation have been written. The coordinator alone writes JSONL, updates cost, and makes stop decisions. Parallel scheduling is deterministic for a given seed, but completion order is allowed to differ. `--jobs > 1` fails early with `--executor delegated`; automatic job selection, provider-specific rate limits, dynamic throttling, distributed execution, and cross-run merging are outside this version.

After `run` completes, use the generated `report.md` as the benchmark result presented to the user. It contains the tested scenarios/task types and the expanded per-dimension score table; do not replace it with a hand-written summary that drops those sections. If `summary.json` reports `reports.status: failed`, surface the report error and stop rather than presenting an incomplete result.

If a terminal disconnect or interruption leaves a partial run, repeat the exact command with `--resume --execute --confirm`, the same `--jobs`, and the same output directory. RoleBench verifies the experiment signature, source workspace/image content fingerprints, skill/harness fingerprint, all four raw/evaluation audit streams, and their independent content digests before rewriting the manifest or starting another provider call. The plan shows completed and remaining candidate/judge cells, preserves run-wide concurrency for latency reporting, identifies the narrower resume segment separately, skips persisted candidate results, and continues only missing calls. Do not use `--resume` after changing the harness, task inputs, source fixture contents, or concurrency width; start a new output directory and rerun the full matrix. See [operations.md](references/operations.md) for the recovery protocol, failure classification, integrity boundary, and OMP compatibility checks.

The default `direct` executor calls the selected model with a role contract prompt and supports bounded parallel cells through `--jobs`. Use `--executor delegated` for the closer-to-production subagent path: it creates a temporary project agent with the candidate model and asks an OMP controller to invoke that agent through the `task` tool. The controller defaults to `modelRoles.default`; `--controller-model provider/model[:thinking]` overrides it, and delegated preflight fails when neither is available. The plan records the resolved controller and includes its estimated requests, tokens, and cost in the total budget gate. Delegated execution remains serial in this version. Parallel latency reflects provider concurrency/load effects, so compare only runs with the same executor and `jobs` setting.

The V1 evaluator deterministically checks declared output contracts, file state, changed-path scope, failure-output patterns, and explicitly approved command results. With `--judge-model`, V2 sends a blinded task/output/evidence packet to a versioned fixed judge and writes `judge_results.jsonl` plus `judge_evaluations.jsonl`; each judge uses a temporary minimal OMP config and empty cwd, disables tools, extensions, skills, and rules, because the packet is the complete evaluation contract, so it must not investigate the temporary workspace as a new task. Malformed or incomplete judge evidence is `unknown` and never overwrites V1 results. Execution status, task outcome, dimension status, and hard gates remain separate. Incomplete execution, evaluator limits, missing evidence, and dimensions without applicable checks are `unknown` with a `null` score; no individual hard gate passes without evidence. Snapshot, artifact, or evaluator failures are recorded without overwriting a completed raw provider result. Add `--recommend` to write a recommendation only after every selected role has the minimum repeated runs, all applicable deterministic hard evidence passes, and the fixed judge fully verifies every requested hard dimension. A hard dimension with no deterministic evaluator remains `unknown` in V1 but does not duplicate the fixed-judge gate; blocked candidates retain their reasons. There is no composite score. Tool-trace reconstruction, host security isolation, and statistical significance remain outside this V2 slice. See [evaluation-v2.md](references/evaluation-v2.md) for the schema and limits.

## Response performance (`omp bench`)

`omp bench` measures raw provider response behavior — TTFT vs decode throughput with p50/p95 — without touching `modelRoles` or any config. Use it alongside RoleBench: RoleBench answers per-role quality and real-task latency/cost; `omp bench` isolates provider-side speed independent of roles.

```bash
omp bench anthropic/claude-opus-4-5 openai/gpt-5.2 --runs 3 --json
```

- Profiles: `chat` (default; balanced, TTFT + tok/s), `prefill` (large cache-busted input; input-processing speed), `generation` (long forced output; sustained decode), `mix` (rotates all three). Tune with `--runs`, `--max-tokens`, `--prefill-bytes`, `--prompt`.
- `--cache` measures independent cold/warm prompt-cache pairs (`--cache-pairs`, `--cache-prefix-file`, `--cache-prefix-bytes`); not supported for `openai-codex-responses`.
- `--par N` sets parallel queries (default 4); `--service-tier` overrides the configured `tier.*` setting per model family; `--json` gives machine-readable output. Selectors accept fuzzy ids (`omp bench opus sonnet`) and thinking suffixes (`provider/model:low`).

Recommended workflow: run `omp bench` as a cheap screening stage to drop unresponsive or slow candidates before committing to a full rolebench matrix, and to attribute observed latency (TTFT vs prefill vs decode vs cache) when a rolebench cell looks slow. It makes metered provider calls but has no cost cap and no `--confirm` gate, so the consent plan must state the models, profile, `--runs`, and `--par` before executing.

## Instruction following (`omp if-bench`)

`omp if-bench` measures instruction following and working memory: each model plays one cached thread of glyph-array actions with a moving cat-sound directive; turn N issues N actions, so later turns stress long-context state tracking. Like `omp bench`, it runs model selectors directly without touching `modelRoles` or any config.

```bash
omp if-bench opus sonnet gpt-5.2 --json
```

- `--turns` caps turns per model (default 24); `--length` sets the character-array length (even, 8–26, default 24); `--nya-max` bounds the accepted cat sound `nya{1,N}` (default 8); `--max-tokens` caps output per turn (default 32768).
- `--par N` benchmarks models concurrently (default 4); use `--par 1` for a clean sequential probe. `--json` emits a machine-readable per-turn transcript.

Use it as a second screening stage after `omp bench`: a model that is fast but loses the array state or the moving directive is a poor fit for `task`/`default` roles that must hold constraints across long sessions. Same consent rule as `omp bench` — metered calls, no cost cap or `--confirm` gate, so state the models, `--turns`, `--length`, and `--par` in the plan before executing.

## Reports

`run` writes `report.md` and `report.json` automatically. Both formats retain the effective concurrency, dependency-aware provider-wave estimate, maximum/peak copied-workspace estimate, cost cap, final budget status, and concurrent in-flight cost exposure alongside the quality evidence. The command below rerenders either format when an existing run needs a refreshed report; without `--output` it prints to stdout, so pass `--output <run>/report.md` to actually replace the file:

```bash
python3 omp-rolebench/scripts/rolebench.py report \
  --run runs/20260828-120000 \
  --format markdown
```

Reports begin with a benchmark-scope table listing each local scenario, task type, workspace kind, difficulty, checks, and target dimensions. The model-summary table puts each tested role/model/thinking variant on one row with execution and outcome counts, every evaluated dimension, gate/recommendation status as overall performance, p50 latency, and observed cost. It does not invent a composite score. Detailed sections retain assignment counts, output size, hard-gate failures, fixed-judge results, candidate-level blocking reasons, and links to raw/evaluation/artifact evidence. Treat cost as unknown unless completed assistant events contain usage/cost data; multi-turn assistant costs are summed, while missing usage remains unknown rather than inferred. See [evaluation-v2.md](references/evaluation-v2.md) for the fixed-judge and recommendation artifacts.

When the data exists, reports append three expanded sections after the aggregate tables: `Task Details` (full prompt, declared expectations/invariants/failure mode per task), `Dimension Details` (per-replicate deterministic and judge status/score with the judge's evidence bullets verbatim; dimensions whose only record is the evaluator's no-evidence placeholder are omitted), and `Outputs` (per-cell latency, cost, and the model output truncated at 2000 characters with an in-fence marker). Long text is fenced with dynamically sized tilde fences so nested fences cannot break the report; the same expansions are mirrored in `report.json` under `task_details`, `dimension_details`, and `outputs`.

## Guardrails

- Keep candidate discovery and task execution snapshots in the run directory for comparability.
- Run vision tasks only when the fixture contains an image and the catalog/preflight says the model supports vision; otherwise record a visible error.
- Do not use a model's generated text as the sole judge of another model. Prefer deterministic invariants and blind human or fixed-judge review for quality.
- Preserve raw stdout/stderr and exit codes. Provider failures, timeouts, and malformed JSON are data, not reasons to silently retry with another model.
- Treat a harness defect as a stop condition, not a candidate failure. Before provider calls start, correct preflight defects and regenerate the plan. After any cell starts, do not edit `SKILL.md`, `scripts/`, `references/`, `assets/`, `agents/`, or the active task suite within that run. The harness fingerprints the skill tree around every new provider cell; normalized tasks are fixed in the manifest signature. Preserve artifacts, report the defect separately, and obtain user authorization for maintenance. After the fix, use a new output directory and rerun the complete matrix so every cell uses the same harness. Reserve `--resume` for interruption with an unchanged harness.
- Treat copied workspaces as isolation for repeatability, not as a host sandbox. Use disposable fixtures and review every command check before enabling it.
- Never remove a cell workspace while its subprocess is still running. Keep it through snapshot, diff, artifact, deterministic evaluation, and approved command checks; remove only the exact harness-created temporary root afterward. Treat an unresolved cleanup path as a harness failure, not as a successful run.
- V1 has three tracked non-blocking limits: approved command capture is read-capped but not write-capped, rerun rotation happens before later gates, and regex-worker crashes are currently indistinguishable from non-matches. See [evaluation-v1.md](references/evaluation-v1.md) before relying on these paths for hostile inputs.
