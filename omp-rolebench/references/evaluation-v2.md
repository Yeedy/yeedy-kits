# Deterministic and Semantic Evaluation V2

RoleBench V2 keeps the V1 evidence chain and adds an optional fixed judge and a gated recommendation pass. V2 does not rewrite `results.jsonl` or `evaluations.jsonl`, and it does not claim that a copied workspace is a host security sandbox.

## Scope

The first V2 slice implements:

1. A versioned fixed-judge rubric and prompt. The judge receives the task, model output, errors, and deterministic evidence, but not the candidate model identity, workspace path, or image path.
2. Separate `judge_results.jsonl` and `judge_evaluations.jsonl` files. Malformed, timed-out, or incomplete judge output is `unknown` and never replaces a deterministic failure.
3. Preflight accounting for judge requests, input/output tokens, and USD cost. The displayed estimate and `--max-cost-usd` gate cover candidate-model, delegated-controller, and judge calls.
4. Repeated-run gating and a recommendation artifact. `--recommend` requires `--judge-model`; every selected role must have at least `--min-recommendation-runs` observations per task, all applicable deterministic hard evidence must pass, and the judge must fully verify every requested hard dimension. A dimension that has no deterministic evaluator remains `unknown` in V1 but does not by itself block a judge-verified recommendation. Ineligible candidates remain visible with blocking reasons.

The recommendation is a constrained ordering, not a single composite quality score. Eligible candidates are ordered by fixed-judge non-hard-dimension score, then p50 latency, then observed cost; missing usage remains `unknown` rather than zero and known costs sort first. Correctness, honesty, and safety remain hard gates and cannot be traded for speed or price.

## CLI

Preview the complete plan, including fixed-judge cost:

```bash
python3 omp-rolebench/scripts/rolebench.py run \
  --models opencode-go/glm-5.3-flash,opencode-go/deepseek-v4-flash-vision-exp \
  --roles smol,task,advisor \
  --mode role-isolated \
  --suite core \
  --runs 3 \
  --judge-model opencode-go/glm-5.3-flash \
  --pricing-file pricing.json \
  --max-cost-usd 0.50 \
  --output runs/v2-preview
```

After reviewing the exact plan, approve the same run and enable recommendations:

```bash
python3 omp-rolebench/scripts/rolebench.py run \
  --models opencode-go/glm-5.3-flash,opencode-go/deepseek-v4-flash-vision-exp \
  --roles smol,task,advisor \
  --mode role-isolated \
  --suite core \
  --runs 3 \
  --judge-model opencode-go/glm-5.3-flash \
  --min-recommendation-runs 3 \
  --pricing-file pricing.json \
  --max-cost-usd 0.50 \
  --execute --confirm --recommend \
  --output runs/v2-20260829
```

`--judge-timeout` controls the per-judge subprocess timeout. When `--recommend` is used without `--judge-model`, the fixed judge defaults to `modelRoles.default` from the OMP config; an explicit `--judge-model provider/model[:thinking]` overrides it. The plan records this source, and changing it during consent requires a new preview and confirmation. Execution with a fixed judge always requires known pricing for the judge; when a cost cap is supplied, pricing must also be known for every candidate and the delegated controller when enabled. Consent is still required for the combined estimate; `--confirm` is not a substitute for showing the plan to the user.

## Judge contract

The current rubric version is recorded as `rolebench-v2-20260831`. The prompt requires JSON with a `dimensions` object. Each requested dimension has:

```json
{
  "status": "pass | fail | unknown",
  "score": 2,
  "evidence": ["observable reason"]
}
```

RoleBench runs the fixed judge with `--no-tools` in an empty temporary working directory. The prompt packet contains the complete task, candidate output, and deterministic evidence needed for evaluation; the judge must not investigate the workspace or run tools. RoleBench reads the final text segment from the last completed assistant turn in the OMP event stream. When a provider emits finalized text through `message_update.assistantMessageEvent.text_end`, that segment takes precedence over a `message_end` payload that may merge earlier tool chatter. It accepts the equivalent k3 layouts observed in real runs: a dimension object, a dimension list keyed by `name` or `dimension`, a requested-dimension map under `evaluation`, or requested dimensions at the top level. These layouts are normalized to the canonical dimension object; duplicate or malformed entries still fail visibly. Scores become `2` for pass, `0` for fail, and `null` for unknown. Missing dimensions, malformed evidence, invalid JSON, non-zero exit, and timeout all become `unknown`. The fixed judge is an external evaluator, not a self-reporting field from the candidate model. Its prompt is versioned in code and its raw stdout/stderr are retained for audit.

## Artifacts and reports

- `judge_results.jsonl`: raw fixed-judge subprocess output, status, stderr, usage, and provider metadata.
- `judge_evaluations.jsonl`: normalized dimension statuses, scores, evidence, hard-gate status, and rubric version.
- `recommendations.json`: schema version, minimum-run requirement, each candidate's eligibility/blockers, and one recommendation record per selected role.
- Markdown and JSON reports include fixed-judge rows and recommendations when those artifacts exist. Markdown recommendation tables retain candidate-level blocking reasons, hard-dimension coverage, and links to raw/evaluation/artifact evidence. They retain V1 outcomes and dimensions separately.

Use the recommendation only when its status is `complete`. A `blocked` status is an actionable result: inspect the candidate-level reasons and the underlying raw/evaluation artifacts rather than treating the absence of a winner as a model failure.

## Remaining V2 prerequisites

These items are intentionally outside the first V2 slice:

- Tool-event and final-answer cross-validation waits for a stable, versioned OMP event schema. The current evaluator does not reconstruct tool traces.
- Stronger easy/hard paired tasks, hidden assertions, controlled failures, prompt-injection/inducement cases, and real side-effect fixtures still need to be added before treating a recommendation as broadly representative.
- Host-level network, credential, process, and filesystem isolation must be provided by an external sandbox before testing dangerous side effects. Copied workspaces and `--auto-approve` are not that boundary.
- Statistical confidence intervals, calibration checks, judge agreement, and judge-bias audits are future work. Repeated runs currently provide a minimum evidence gate and p50/observed-cost context, not a significance claim.

Fixed-judge calls add cost and can introduce evaluator bias. Keep the rubric version, judge model, pricing, prompt inputs, and raw evidence with each run so a later review can reproduce the decision.
