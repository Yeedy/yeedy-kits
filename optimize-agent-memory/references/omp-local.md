# OMP Local Memory

Use this route only when Oh My Pi has `memory.backend: local`.

## Establish Scope

1. Confirm the active OMP version, profile, and agent root. Resolve the root from `PI_CODING_AGENT_DIR` when set; otherwise use the named `OMP_PROFILE` or `PI_PROFILE` under `PI_CONFIG_DIR` (default `~/.omp`), falling back to `~/.omp/agent`.
2. Default to every project scope stored under the active profile. Narrow to a named project or scope only when the user explicitly requests it.
3. Confirm `memory.backend: local`; confirm `autolearn.enabled: true` when lesson capture is in scope.
4. Treat `local` as local storage, not necessarily local inference. Phase 1 normally uses the `default` model role and Phase 2 uses `smol` unless an explicitly supported memory-model setting overrides them.

For an explicit project scope, resolve the exact scope through OMP's public memory surface from that project's working directory or through trustworthy runtime metadata. If it cannot be resolved uniquely, report the ambiguity and stop that project audit; do not choose a scope by basename similarity.

## Read

Start with `scripts/memory_baseline.py --runtime omp` to enumerate stored project scopes and host-wide job health. By default, audit the artifacts reported for every scope. For an explicitly selected project, use the baseline only to locate and diagnose that scope, then restrict content inspection and recommendations to it. Use OMP's public surfaces for the active project only when runtime activation is authorized, so project identity remains correct:

1. Read `memory://root` for the compact memory payload resolved for the requesting session.
2. Read `memory://root/MEMORY.md` for consolidated memory.
3. Read `memory://root/learned.md` for explicit and autolearn lessons.
4. Read relevant `memory://root/skills/<name>/SKILL.md` artifacts when generated skills add duplication or bad guidance.
5. Use `/memory stats` and `/memory diagnose` when interactive slash commands are available. Ask the user to run them only when their output is needed for a concrete diagnosis.

Treat CLI access through a fresh OMP process as runtime activation unless the installed version establishes that it cannot write persistent state. In particular, `omp read memory://root` can initialize the local-memory schema and job runtime even though it returns read output. Runtime activation can write SQLite and generated artifacts, reclaim or enqueue memory jobs, and call the configured memory models. Under an explicitly read-only boundary that forbids database writes or paid calls, do not invoke it; inspect baseline artifacts and read-only job metadata, then report injection as unverified.

The `memory://` surface is bound to the requesting session, not merely its working directory. Concurrent sessions in the same directory must not be used interchangeably for verification. Subagents and Advisor access the memory of their owning main session; they are not independent project-verification contexts. Record the verified session identity when available and its workspace. A check in a newly authorized session verifies that session only, not an earlier session in the same directory.

If a session has expired and the request is rejected, report the session failure and injection as unverified. Do not interpret the rejection as missing memory or Stage 2 failure, or substitute another live session as proof that the original session works. Session-bound routing does not establish separate persistent stores or remove runtime initialization side effects.

For other stored scopes, inspect only the read-only artifact roots and job metadata reported by the baseline. Do not decode encoded scope-directory names or guess their original working directories. If trustworthy runtime metadata resolves a scope to an existing working directory, use an authorized session in that directory with OMP's public memory surface for the injection check. If a scope cannot be mapped safely, report the scope ID and artifact state without inventing a project name.

If no memory exists, distinguish a new setup from failure: local consolidation processes persisted sessions at startup, while recent, active, old, or ephemeral sessions may be ineligible.

Distinguish pipeline stages:

- Stage 1 success produces `raw_memories.md` and `rollout_summaries/`; these are intermediate evidence, not proof that memory is injected.
- Stage 2 success produces `MEMORY.md`, `memory_summary.md`, and optional generated skills.
- Treat a successful `memory://root` read in the target session as the injection-surface check. If it reports no artifacts while Stage 1 files for the same scope exist, diagnose Stage 2 instead of curating content blindly.

Use the baseline for read-only job and artifact metrics across every stored scope. Prefer `/memory diagnose` for the active scope when available. When deeper local diagnosis is needed, read job status from the OMP database in read-only mode; never update job rows. Surface `last_error` exactly.

If Stage 2 reports a model-output or JSON parse failure, allow the existing retry budget to run first. If it repeats, propose testing a different `smol` role model with reliable structured output, one change at a time. Do not change memory size, concurrency, lease, or retry settings to mask a model-output failure.

Check scope quality before content quality for every stored scope. A broad working directory such as Desktop can combine unrelated sessions into one project memory. Recommend starting OMP in the actual project root; do not move or merge existing scope data without explicit approval.

## Runtime Refresh And Recovery

Keep runtime recovery separate from the read-only audit:

1. Use the baseline to identify each exact workspace, job kind and key, current status, retry budget, and `last_error`. Do not infer a workspace from an encoded memory-directory name.
2. Before activation, disclose the exact workspace scopes, OMP-owned SQLite and generated-artifact writes, possible model calls and cost, and any known risk that a host-wide worker could touch another queued scope. Wait for the user's direct approval of that boundary.
3. Start OMP from one approved workspace at a time. Keep the process alive until its targeted jobs reach `done` or `error`; closing it while a job is `running` can leave an expired lease that requires another startup to reclaim. Do not run recoveries concurrently against the shared database.
4. Monitor job state through the read-only baseline. If an unapproved scope starts processing, stop further activation and report the expanded effect. Do not update job rows, leases, retry counters, or ownership fields directly.
5. Stop after one native attempt per approved failing job. If the same error returns, report it and propose the smallest evidence-backed next change rather than consuming the remaining retry budget automatically.
6. After all targeted jobs reach a terminal state, close the OMP process and re-run the baseline. Verify `memory://root` from each approved workspace only when that activation is covered by the current authorization. A later read-only scheduled audit should observe completed recovery through stored artifacts and job metadata, not reactivate OMP to repeat verification. Record any scope that remains unverified.

## Audit Efficiency

Estimate injected size from `memory://root` and label it approximate. The baseline reports artifact `bytes` and `lines` as size proxies, not token counts; use the actual `memory://root` payload for the estimate. Check whether low-value lessons crowd out useful summary content under `memories.summaryInjectionTokenLimit`.

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

Re-read edited lessons and compare the baseline. Run stats, diagnostics, or `memory://root` from a new session after refresh only when runtime activation is authorized. Under a read-only boundary, report artifact availability and size separately from actual injection, mark injection unverified, and complete the audit without requesting write permission or retrying rejected activation. A `learn` call affects future sessions rather than mutating the active prompt prefix.
