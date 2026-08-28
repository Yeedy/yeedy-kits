---
name: omp-rolebench
description: Benchmark Oh My Pi model-role assignments. Use when comparing all models, models from selected providers, or explicitly named models across all or selected OMP roles, generating isolated/all-same assignment matrices, running repeatable role tasks, or producing a cost/quality report.
---

# OMP RoleBench

## Overview

Use `scripts/rolebench.py` to discover the current OMP model catalog, resolve `modelRoles` from an OMP config, generate bounded assignment experiments, optionally execute tasks through OMP, and write JSONL results plus a Markdown report. The harness never edits the user's OMP config; execution is opt-in because it can call metered providers.

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

## Consent and budget gate

Before any provider call, show the user a short plan containing the candidate models, roles, assignment mode, executor, task count, repeat count, request count, estimated input/output tokens, and estimated USD cost. Ask for explicit approval in the conversation. If the user states a budget in natural language (for example, "benchmark cost must stay below $0.50"), convert it to `--max-cost-usd 0.50` and repeat the cap in the confirmation request.

The command also enforces the gate: `--execute` without `--confirm` writes the manifest and estimate but exits without making requests. Only pass `--confirm` after the user approves that exact plan. A supplied `--max-cost-usd` requires known prices from `--pricing-file` or catalog metadata; an unknown estimate, or an estimate above the cap, blocks execution. Pricing files use USD per million tokens:

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

`--tasks-file` accepts JSON or JSONL task fixtures with `id`, `role`, `prompt`, and optional `workspace`, `expected`, and `max_output_tokens`. When it is omitted, choose between `--task-source core` and `--task-source recent`; the CLI default remains `core` for backward compatibility. At the skill interaction layer, if the user has not chosen a source, ask whether to use the generic suite or recent local work before generating the plan.

Use `--task-source recent` to generate one task per selected role and workspace from recent Git commits and working-tree changes. Repeat `--recent-workspace` for multiple workspaces, or pass comma-separated paths. Set `--recent-days` and `--recent-commits` to control the lookback window. Git paths are scoped and normalized relative to each selected workspace; tracked deletions remain context only, and only existing image files become vision fixtures. For a non-Git workspace, the generator falls back to recently modified files. If Git is present but a command fails, the generator records the error and shows a warning in the plan instead of hiding it. It records commit subjects and file paths in the generated prompts and manifest, but does not read OMP sessions, Codex memory, or past chat history. Each generated task keeps its workspace path so every benchmark cell runs against an isolated copy; review `manifest.json` before consenting to execution.

If `--tasks-file` is provided, it takes precedence over `--task-source`.

`--execute` invokes `omp -p --mode json --no-session` and appends each result immediately to `results.jsonl`. Each cell gets a temporary config overlay and workspace; the user's normal config is never changed. Reusing an output directory rotates an existing `results.jsonl` to a timestamped `results.jsonl.previous-*` file before writing the new manifest. Without `--execute`, `manifest.json` and the cost estimate are still produced so the experiment can be reviewed before making provider calls.

The default `direct` executor calls the selected model with a role contract prompt. Use `--executor delegated` for the closer-to-production subagent path: it creates a temporary project agent with the candidate model and asks an OMP controller to invoke that agent through the `task` tool. Delegated runs add controller overhead to latency, so compare like with like.

The default executor measures completion, latency, output length, and process errors. It does not pretend to provide a semantic quality score. Add task-specific scores later through the `scores` field in `results.jsonl` or an external reviewer.

## Reports

```bash
python3 omp-rolebench/scripts/rolebench.py report \
  --run runs/20260828-120000 \
  --format markdown
```

Reports show assignment counts, pass/error rates, latency, output size, thinking variants, and per-role/per-model rows. Treat cost as unknown unless the provider returns usage/cost data; unknown cost is reported as unknown rather than inferred.

## Guardrails

- Keep candidate discovery and task execution snapshots in the run directory for comparability.
- Run vision tasks only when the fixture contains an image and the catalog/preflight says the model supports vision; otherwise record a visible error.
- Do not use a model's generated text as the sole judge of another model. Prefer deterministic invariants and blind human or fixed-judge review for quality.
- Preserve raw stdout/stderr and exit codes. Provider failures, timeouts, and malformed JSON are data, not reasons to silently retry with another model.
