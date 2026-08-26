# OMP Local Memory

Use this route only when Oh My Pi has `memory.backend: local`.

## Establish Scope

1. Confirm the active OMP version, profile, working directory, and intended project.
2. Confirm `memory.backend: local`; confirm `autolearn.enabled: true` when lesson capture is in scope.
3. Treat `local` as local storage, not necessarily local inference. Phase 1 normally uses the `default` model role and Phase 2 uses `smol` unless an explicitly supported memory-model setting overrides them.

## Read

Use OMP's public surfaces so project identity and profile scoping remain correct:

1. Read `memory://root` for the compact payload injected into a new session.
2. Read `memory://root/MEMORY.md` for consolidated memory.
3. Read `memory://root/learned.md` for explicit and autolearn lessons.
4. Read relevant `memory://root/skills/<name>/SKILL.md` artifacts when generated skills add duplication or bad guidance.
5. Use `/memory stats` and `/memory diagnose` when interactive slash commands are available. Ask the user to run them only when their output is needed for a concrete diagnosis.

Do not reimplement OMP's encoded project path when `memory://` resolves the active project. If no memory exists, distinguish a new setup from failure: local consolidation processes persisted sessions at startup, while recent, active, old, or ephemeral sessions may be ineligible.

Distinguish pipeline stages:

- Stage 1 success produces `raw_memories.md` and `rollout_summaries/`; these are intermediate evidence, not proof that memory is injected.
- Stage 2 success produces `MEMORY.md`, `memory_summary.md`, and optional generated skills.
- Treat `memory://root` as the injection check. If it reports no artifacts while Stage 1 files exist, diagnose Stage 2 instead of curating content blindly.

Run `scripts/memory_baseline.py --runtime omp` for read-only job and artifact metrics. Prefer `/memory diagnose` when available. When deeper local diagnosis is explicitly in scope, read job status from the OMP database in read-only mode; never update job rows. Surface `last_error` exactly.

If Stage 2 reports a model-output or JSON parse failure, allow the existing retry budget to run first. If it repeats, propose testing a different `smol` role model with reliable structured output, one change at a time. Do not change memory size, concurrency, lease, or retry settings to mask a model-output failure.

Check scope quality before content quality. A broad working directory such as Desktop can combine unrelated sessions into one project memory. Recommend starting OMP in the actual project root; do not move or merge existing scope data without explicit approval.

## Audit Efficiency

Estimate injected size from `memory://root` and label it approximate. Check whether low-value lessons crowd out useful summary content under `memories.summaryInjectionTokenLimit`.

Recommend tuning only for observed evidence:

| Signal | Setting | Direction |
| --- | --- | --- |
| Useful memory is truncated | `memories.summaryInjectionTokenLimit` | Increase carefully |
| Injection is large and low-value | content first, then `memories.summaryInjectionTokenLimit` | Curate before lowering |
| Eligible sessions outpace startup processing | `memories.maxRolloutsPerStartup`, `memories.stage1Concurrency` | Increase within model and rate limits |
| Important sessions age out | `memories.maxRolloutAgeDays` | Increase |
| Active work is extracted prematurely | `memories.minRolloutIdleHours` | Increase |
| Extraction is too costly or slow | supported memory-model or model-role configuration | Compare quality before switching |

Do not tune lease, retry, heartbeat, payload, or fallback-token settings without a matching diagnostic symptom.

## Apply

1. Edit only approved entries in `learned.md`. Preserve unrelated bullets, headings, prose, ordering, and final newline.
2. Use `learn` for a new durable lesson when no old item must be removed or rewritten. Do not use it to simulate a merge because superseded bullets remain.
3. Do not hand-edit `MEMORY.md`, `memory_summary.md`, `raw_memories.md`, rollout summaries, or generated memory skills; consolidation owns them.
4. For stale generated memory, correct a durable source when possible, then have the user run `/memory rebuild` or `/memory enqueue`. Processing occurs on a subsequent writable, persisted OMP startup.
5. Treat `/memory clear` and `/memory reset` as destructive.
6. Do not use `recall`, `retain`, `reflect`, or `memory_edit`; the local backend does not expose those structured-memory tools.
7. Never edit OMP's SQLite state directly.

## Verify

Re-read edited lessons, compare the baseline, run available stats or diagnostics after refresh, and inspect `memory://root` from a new session. A `learn` call affects future sessions rather than mutating the active prompt prefix.
