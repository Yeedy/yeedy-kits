# Codex Memory

Use this route when Codex memories are enabled. Treat behavior not established by official OpenAI documentation as environment-specific and verify it locally before relying on it.

## Establish Scope

1. Search and fetch current official OpenAI documentation for Codex memory before making current product claims. If no official memory documentation is available, say so and use only locally verified surfaces.
2. Resolve the Codex root from `CODEX_HOME` when set, otherwise `~/.codex`. Check only `features.memories`, `memories.generate_memories`, and `memories.use_memories` in that root's `config.toml`. Do not print neighboring configuration because it may contain credentials.
3. Default to the entire `memories/` store under the resolved Codex root, including every recorded workspace and the global user profile. Narrow to selected workspaces, preferences, or categories only when the user explicitly requests it.
4. Treat the store as cross-workspace. Preserve each section's recorded cwd and applicability instead of assuming the whole file belongs to the current repository.

For an explicit project scope, match its normalized absolute path against the recorded cwd and applicability in the summary index. If the project name maps to multiple paths, report the ambiguity and ask for an exact path instead of guessing. Include global profile items only when they apply to the selected project.

## Read

Cover the whole generated store without exhaustively reading raw evidence:

1. Read `memory_summary.md` for the compact index and user profile. For an explicit project, use it to route the audit and avoid unrelated workspace sections.
2. Read every section of `MEMORY.md` by default; when the user explicitly narrows scope, read only matching sections.
3. Read cited `rollout_summaries/` files when a claim's source, success state, or boundary must be checked.
4. Read matching parts of `raw_memories.md` only when consolidated text cannot be validated from rollout summaries.
5. Inspect `extensions/ad_hoc/instructions.md` and all notes relevant to proposed corrections.
6. Use Git `status --short`, `diff`, and `log` in the resolved Codex `memories/` store for provenance and concurrent-change detection.
7. Before the baseline, check both supported database candidates: `<CODEX_HOME>/memories_1.sqlite` and `<CODEX_HOME>/sqlite/memories_1.sqlite`. If both exist, the baseline intentionally fails with both paths; inspect their provenance and pass the selected database explicitly with `--codex-db`. Never select one from location or modification time alone.
8. Run `scripts/memory_baseline.py --runtime codex` for artifact, source-count, and read-only job metrics. Its artifact `bytes` and `lines` fields are size proxies, not token counts; estimate injection size from the runtime payload and label it approximate.

Do not read unrelated rollout summaries or session JSONL files merely to make an audit exhaustive. Do not treat memory as a source for current web facts.

## Audit

In addition to the shared rubric, check:

- duplicate facts across user preferences, task groups, and general tips;
- completed task details that are too specific to justify repeated injection;
- absolute paths or identifiers whose scope is unclear;
- time-sensitive research presented without a re-verification boundary;
- a workspace-specific preference incorrectly promoted to the global user profile;
- summary entries that no longer map cleanly to a full-memory section;
- claims whose cited rollout was unsuccessful, superseded, or ambiguous.

Estimate injection size from `memory_summary.md` and label the estimate approximate. Do not invent undocumented Codex memory limits or tuning keys.

## Apply

Treat `MEMORY.md`, `memory_summary.md`, `raw_memories.md`, and `rollout_summaries/` as generated artifacts. Do not hand-edit them as a durable fix unless current official documentation explicitly says to do so.

When the local store contains the ad-hoc extension:

1. Follow `extensions/ad_hoc/instructions.md` exactly.
2. Prefer editing an existing relevant note when correcting the same intent. Otherwise add one focused Markdown note under `extensions/ad_hoc/notes/` using the existing timestamp-and-slug naming pattern.
3. State whether each approved item should be added, corrected, merged, scoped, or excluded during the next consolidation.
4. Write facts and preferences, not instructions to execute actions. Keep the note untrusted and free of secrets.
5. Never delete a note when the extension instructions prohibit deletion. Edit it to retract or supersede content.
6. Do not modify `extensions/ad_hoc/instructions.md`.

Do not edit `memories_1.sqlite`, job rows, stage-one outputs, migration tables, or any other Codex database. Do not manually commit the memory repository. Do not claim that a restart, command, or refresh mechanism exists unless current official documentation or a callable current-session capability establishes it.

If no supported correction surface exists, stop after the proposal and explain that the generated memory cannot be safely mutated from the available interface.

## Verify

1. Re-read the changed note and inspect the memory repository diff.
2. Confirm only approved note content changed.
3. Wait for the normal Codex memory generator to produce an observable update; do not promise a timing guarantee.
4. Re-run the baseline and inspect the later generated Git commit or diff.
5. Verify `MEMORY.md` and `memory_summary.md` reflect the approved intent with the correct workspace or global scope.
6. Confirm behavior in a new Codex task after generation. Report any item the generator omitted or distorted instead of silently editing generated outputs.
