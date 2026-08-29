---
name: omp-rolebench
description: Benchmark Oh My Pi model-role assignments. Use when comparing all models, models from selected providers, or explicitly named models across all or selected OMP roles, generating isolated/all-same assignment matrices, running repeatable role tasks, or producing a cost/quality report.
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

Before any provider call, show the user a short plan containing the candidate models, roles, assignment mode, executor, task count, repeat count, request count, estimated input/output tokens, estimated USD cost, deterministic evaluation dimensions, and any exact command-check argv. When a fixed judge is enabled, include the judge model, whether it came from `modelRoles.default` or an explicit `--judge-model` override, rubric version, request count, token estimate, and cost; the displayed total and `--max-cost-usd` gate cover both candidate and judge calls. Ask for explicit approval in the conversation. Before executing, the user can name a different judge model; rerun the preview with `--judge-model provider/model[:thinking]` and obtain consent for that revised plan. If the user states a budget in natural language (for example, "benchmark cost must stay below $0.50"), convert it to `--max-cost-usd 0.50` and repeat the cap in the confirmation request.

The command also enforces the gate: `--execute` without `--confirm` writes the manifest and estimate but exits without making requests. Only pass `--confirm` after the user approves that exact plan. Task fixtures with command checks require a separate `--allow-check-command` flag after the displayed argv has been reviewed; the commands are trusted local code, not a sandbox. A fixed judge always requires a known judge price before execution, even without a cap. `--recommend` without `--judge-model` resolves the judge from `modelRoles.default` and fails early if that role is not configured. A supplied `--max-cost-usd` additionally requires known prices for every candidate and the judge; an unknown estimate, or an estimate above the cap, blocks execution. Pricing files use USD per million tokens:

```json
{
  "opencode-go/glm-5.3-flash": {
    "input_per_million": 0.075,
    "output_per_million": 0.25
  }
}
```

The estimate is a preflight budget, not a promise about tool-heavy context growth. During execution, provider-reported costs are accumulated and later requests stop when the cap is reached. If an attempted provider call returns no cost, including a timeout, a capped run stops after that call with `budget_status: unverifiable`; a single in-flight request can still overshoot the cap.

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

`--tasks-file` accepts task schema V1 as JSON or JSONL. Tasks require `id`, `role`, and `prompt`; they may declare difficulty metadata, easy/hard pairing, expected output, workspace invariants, failure modes, and target dimensions. See [evaluation-v1.md](references/evaluation-v1.md) for the complete schema and evaluator behavior. When a tasks file is omitted, choose between `--task-source core` and `--task-source recent`; the CLI default remains `core` for backward compatibility. At the skill interaction layer, if the user has not chosen a source, ask whether to use the generic suite or recent local work before generating the plan.

The generic core suite remains a low-cost smoke suite. Its deterministic protocol and workspace tasks provide usable V1 checks, while subjective roles remain `unknown` where evidence is insufficient. For role-allocation decisions, provide paired easy/hard fixtures plus controlled failure or adversarial cases rather than treating core-suite latency as quality.

Use `--task-source recent` to generate one task per selected role and workspace from recent Git commits and working-tree changes. Repeat `--recent-workspace` for multiple workspaces, or pass comma-separated paths. Set `--recent-days` and `--recent-commits` to control the lookback window. Git paths are scoped and normalized relative to each selected workspace; tracked deletions remain context only, and only existing image files become vision fixtures. For a non-Git workspace, the generator falls back to recently modified files. If Git is present but a command fails, the generator records the error and shows a warning in the plan instead of hiding it. It records commit subjects and file paths in the generated prompts and manifest, but does not read OMP sessions, Codex memory, or past chat history. Each generated task keeps its workspace path so every benchmark cell runs against an isolated copy; review `manifest.json` before consenting to execution.

If `--tasks-file` is provided, it takes precedence over `--task-source`.

`--execute` invokes `omp -p --mode json --no-session` and appends each raw result immediately to `results.jsonl`. For OMP event streams, the harness records only the last completed assistant text as model output and sums usage from every completed assistant message in the cell; user prompts and tool results are not treated as answers or costs. It then writes the independent quality result to `evaluations.jsonl`. Each cell gets a temporary config overlay and workspace; the user's normal config is never changed. Before/after hashes, changed paths, and small UTF-8 text patches are retained under `artifacts/<run-id>/`. Reusing an output directory rotates existing result and evaluation JSONL files before writing the new manifest. Without `--execute`, `manifest.json` and the combined cost estimate are still produced so the experiment can be reviewed before making provider calls.

The default `direct` executor calls the selected model with a role contract prompt. Use `--executor delegated` for the closer-to-production subagent path: it creates a temporary project agent with the candidate model and asks an OMP controller to invoke that agent through the `task` tool. Delegated runs add controller overhead to latency, so compare like with like.

The V1 evaluator deterministically checks declared output contracts, file state, changed-path scope, failure-output patterns, and explicitly approved command results. With `--judge-model`, V2 sends a blinded task/output/evidence packet to a versioned fixed judge and writes `judge_results.jsonl` plus `judge_evaluations.jsonl`; malformed or incomplete judge evidence is `unknown` and never overwrites V1 results. Execution status, task outcome, dimension status, and hard gates remain separate. Incomplete execution, evaluator limits, missing evidence, and dimensions without applicable checks are `unknown` with a `null` score; no hard gate passes without evidence. Snapshot, artifact, or evaluator failures are recorded without overwriting a completed raw provider result. Add `--recommend` to write a recommendation only after every selected role has the minimum repeated runs and both deterministic and fixed-judge hard gates pass; blocked candidates retain their reasons. There is no composite score. Tool-trace reconstruction, host security isolation, and statistical significance remain outside this V2 slice. See [evaluation-v2.md](references/evaluation-v2.md) for the schema and limits.

## Reports

```bash
python3 omp-rolebench/scripts/rolebench.py report \
  --run runs/20260828-120000 \
  --format markdown
```

Reports show assignment counts, execution pass/error rates, latency, output size, thinking variants, task outcomes, per-dimension evidence status, hard-gate failures, fixed-judge results, and gated recommendations when requested. Recommendation reports include candidate-level blocking reasons, evaluated hard dimensions, and links to raw/evaluation/artifact evidence. Treat cost as unknown unless completed assistant events contain usage/cost data; multi-turn assistant costs are summed, while missing usage remains unknown rather than inferred. See [evaluation-v2.md](references/evaluation-v2.md) for the fixed-judge and recommendation artifacts.

## Guardrails

- Keep candidate discovery and task execution snapshots in the run directory for comparability.
- Run vision tasks only when the fixture contains an image and the catalog/preflight says the model supports vision; otherwise record a visible error.
- Do not use a model's generated text as the sole judge of another model. Prefer deterministic invariants and blind human or fixed-judge review for quality.
- Preserve raw stdout/stderr and exit codes. Provider failures, timeouts, and malformed JSON are data, not reasons to silently retry with another model.
- Treat copied workspaces as isolation for repeatability, not as a host sandbox. Use disposable fixtures and review every command check before enabling it.
- V1 has three tracked non-blocking limits: approved command capture is read-capped but not write-capped, rerun rotation happens before later gates, and regex-worker crashes are currently indistinguishable from non-matches. See [evaluation-v1.md](references/evaluation-v1.md) before relying on these paths for hostile inputs.
