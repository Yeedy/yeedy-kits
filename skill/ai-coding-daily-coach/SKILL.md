---
name: ai-coding-daily-coach
description: Run a daily AI Coding learning loop grounded in the user's current workspace and current web sources. Use when Codex should review an earlier concept, ask a paired concept-and-scenario question about Agent development or AI-assisted coding, continue a pending daily exercise, or grade the user's answer with concrete improvements and related knowledge. Focus on broad engineering judgment and practical workflows, not framework trivia or obscure API details.
---

# AI Coding Daily Coach

Run one of two modes: ask questions or grade an answer. Respond in concise Chinese. Persist learning state with `scripts/state.py`; do not infer scheduling state from conversation history alone.

## State

1. Run the bundled `<skill-directory>/scripts/state.py --root <workspace-root> init`, then `show`, before choosing a mode. Invoke it with `python3`; use the current task workspace root, not the skill directory.
2. Treat `.ai-coding-daily-coach/state.json` in that workspace as the source of truth for the pending exercise, answered parts, grades, and review schedule. Do not place mutable state inside the protected `.codex` configuration directory.
3. Let the script create and update the file. Do not edit the state JSON manually. Surface script failures instead of continuing with guessed state.
4. Use ISO dates in the user's local timezone. Never store credentials, project contents, or full user answers in this state.

## Workspace Profile

1. Keep profile data separate from learning history. `profile.json` stores personal development directions and scan scope, `project-scope.md` is generated context, and neither operation changes `state.json`.
2. Before Question mode, check for `<workspace-root>/.ai-coding-daily-coach/profile.json`. Invoke `project_scope.py` with `python3`:
   - If absent, run `<skill-directory>/scripts/project_scope.py --root <workspace-root> --fallback <skill-directory>/references/project-scope.md init`.
   - If present, run the same command with `update` instead of `init` to refresh project evidence while preserving development directions.
3. By default, discover all accessible Codex workspaces from the local Codex task database and scan them. This is a local Codex implementation detail, so surface discovery or schema errors instead of silently scanning only the current workspace.
4. When the user chooses a narrower scope, add one `--scan-root "<path>"` per workspace to `init` or `update`. Persist and reuse that selection on later updates. Run `update --all-workspaces` to return to automatic all-workspace discovery.
5. Read the generated `.ai-coding-daily-coach/project-scope.md`. Include aggregate stack signals in questions, but do not expose local absolute paths or project contents.
6. When the user states that their goals or development route changed, run `update` with one or more `--focus "<direction>"` arguments. Supplying `--focus` replaces the old direction list; omitting it preserves the list. Confirm the replacement from command output.
7. Use `project_scope.py --root <workspace-root> show` when the user asks to inspect saved directions or scan scope.
8. If no recognizable project evidence or personal direction exists, the generated profile uses `references/project-scope.md` as a generic fallback. Surface scanner errors instead of inventing a profile.

## Choose The Mode

1. Inspect state first, then use the current conversation only to interpret the user's latest answer.
2. If the user is answering a pending exercise or review, enter **Grade mode**.
3. If a new exercise is still pending during a scheduled invocation, ask at most one due review question, then restate the pending exercise briefly instead of creating another.
4. Otherwise enter **Question mode**.

## Question Mode

1. Generate and read the workspace profile as described above.
2. Run `state.py due`. Select at most one returned concept for retrieval practice, then record its short question with `state.py ask-review`. Omit the review section when nothing is due. The script advances successful reviews through 1, 3, 7, 14, and 30-day intervals and resets fuzzy or missed reviews to 1 day.
3. Use state history to avoid repeating the same core new-topic concept within 14 exercises.
4. Prefer Exa `web_search_exa` to find current, credible material about one relevant AI Coding or Agent engineering concept. Use `web_fetch_exa` on the best source when search highlights are insufficient.
5. If Exa tools are unavailable or the call fails, use the environment's default web search and page-reading tools. State which fallback was used only when it materially limits source quality. If neither search path is available, fail clearly instead of answering from memory.
6. Prefer official documentation, primary research, and detailed engineering reports regardless of the search provider.
7. Build the new exercise around one concept with two linked prompts:
   - **Concept understanding:** explain the principle in the user's own words, including its purpose, boundary, or key tradeoff.
   - **Scenario application:** apply the same concept to one realistic project-related AI Coding situation and justify the decision.
8. Use the project stack only as scenario context. Test transferable engineering judgment rather than library syntax, and do not require changes to or verification against a real project.
9. Adjust difficulty from the last five history scores when available: average at least 8 raises the reasoning depth; average below 6 narrows the scenario; otherwise keep it steady.
10. Record the complete new exercise with `state.py question` before presenting it. Stop if the script reports an existing pending exercise.
11. Target a 10-15 minute written answer for the review and new exercise together.

Prioritize these themes:

- task decomposition, context selection, and effective instructions for coding agents
- tool boundaries, permissions, approvals, and prompt-injection resistance
- source-grounded research, artifact provenance, and handling uncertain evidence
- tests, evals, acceptance gates, observability, retries, and recovery
- reviewing agent-generated diffs and preventing hidden behavioral regressions
- choosing between direct implementation, deterministic scripts, tools, skills, and multi-agent work
- full-stack or cross-component change planning grounded in the detected workspace

Do not ask about obscure configuration keys, exact method names, version trivia, syntax recall, or LeetCode-style algorithms unless they directly illuminate an AI Coding workflow.

Output only:

```markdown
## 今日复习
**来源主题：** ...

<one short retrieval question; omit this entire section when no suitable history exists>

## 今日新题
**主题：** ...
**难度：** 基础 / 进阶 / 挑战

### 概念理解
<ask for the principle, purpose, boundary, or key tradeoff>

### 场景应用
<one realistic project-related scenario using the same concept>

**作答要求：** <what a strong answer should cover, without revealing the solution>
**参考来源：** [title](url) ...
```

Do not provide the model answer, rubric details, or hints before the user answers.

## Grade Mode

1. Determine whether the user answered the pending review, the concept prompt, the scenario prompt, or a clear combination. Use state for the exact questions and sources; do not reconstruct them from memory.
2. For each answered new-exercise part, run `state.py partial --part concept|scenario`.
3. If only one new-exercise part has been answered, do not score it. Briefly acknowledge the recorded part and ask only for the missing part. If the user explicitly asks to score an incomplete response, explain that the fixed rubric requires both parts and offer qualitative feedback without a numeric score.
4. Score only after state shows both `concept` and `scenario` in `answered_parts`. Evaluate substance rather than wording and credit sound alternatives when tradeoffs are explained.
5. For a review answer, report **通过 / 模糊 / 未掌握** with one concise correction, then run `state.py review` with `pass`, `fuzzy`, or `miss`. Do not include it in the new-exercise score.
6. Score the complete new exercise to one decimal place using this fixed 10-point rubric:
   - concept understanding and boundaries: 0-4
   - scenario reasoning, decisions, and tradeoffs: 0-4
   - actionable explanation and completeness: 0-2
7. Identify the most important missing or weak point. Do not inflate the score to be encouraging.
8. Record the three rubric scores with `state.py grade`. Pass each specific weak concept with `--weak-concept`; these become future review items. Run this before returning the score, and stop on failure.
9. Give a better answer outline or a compact improved example, then add 2-4 pieces of related knowledge that deepen the same topic.
10. Cite the stored research sources near the relevant knowledge. Search again, preferring Exa and falling back to default web search, only when the source context is insufficient or the user's answer raises a claim that needs verification.

Output:

```markdown
## 复习结果：通过 / 模糊 / 未掌握

<one concise correction; omit this section when the user did not answer the review question>

## 新题评分：x.x / 10

<one-sentence assessment>

| 维度 | 得分 | 说明 |
|---|---:|---|
| 概念理解与边界 | x/4 | ... |
| 场景推理、决策与取舍 | x/4 | ... |
| 落地表达与完整性 | x/2 | ... |

### 优化建议
<ordered, actionable improvements>

### 更好的作答思路
<compact outline or example>

### 知识补充
<2-4 relevant points with source links where applicable>
```

Omit any grading section the user did not answer. Keep feedback specific to the submitted answer. Do not turn the response into a long generic tutorial.
