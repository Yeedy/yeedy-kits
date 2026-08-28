#!/usr/bin/env python3
"""Small, dependency-free benchmark harness for OMP model roles.

The harness is intentionally conservative: discovery and matrix generation are
offline by default; provider calls happen only with --execute.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


DEFAULT_ROLES = (
    "slow",
    "plan",
    "vision",
    "designer",
    "advisor",
    "smol",
    "task",
    "commit",
    "tiny",
    "default",
)
THINKING_LEVELS = {"off", "minimal", "low", "medium", "high", "xhigh", "max", "auto"}
ROLE_PROMPTS = {
    "tiny": "Answer the request directly in one concise, correct response.",
    "smol": "Perform this small, bounded operation. Return the requested result and no unrelated discussion.",
    "commit": "Inspect the supplied change context and propose a conventional commit message covering every included change.",
    "task": "Complete the task in the workspace. Inspect relevant files, make only necessary changes, and report what was verified.",
    "slow": "Analyze the task carefully. State the root cause, alternatives, risks, and a concrete recommendation.",
    "plan": "Produce an implementation plan with ordered steps, affected files or boundaries, verification, and rollback risks.",
    "vision": "Inspect the supplied image and describe only visible, actionable facts. Do not invent details that are not visible.",
    "designer": "Review the supplied interface or product requirement and give concrete hierarchy, interaction, and accessibility improvements.",
    "advisor": "Review the proposal or code context critically. Identify correctness, security, reliability, and maintainability risks first.",
    "default": "Handle the request as the primary OMP assistant while preserving scope and verifying important claims.",
}
CORE_TASKS = [
    {"id": "tiny-001", "role": "tiny", "prompt": "Return the JSON object {\"answer\": 42} and nothing else."},
    {"id": "smol-001", "role": "smol", "prompt": "Summarize in three bullets: a cache hit avoids recomputing a stable prefix, but stale invalidation can return old data."},
    {"id": "commit-001", "role": "commit", "prompt": "Given the change: add provider filtering, validate empty model selectors, and write a JSONL result. Suggest a commit message."},
    {"id": "task-001", "role": "task", "prompt": "In the workspace, create a file named benchmark-note.txt containing exactly the word PASS, then verify it."},
    {"id": "slow-001", "role": "slow", "prompt": "Explain the likely root cause when a benchmark's p95 latency rises but p50 stays flat. Give two tests to distinguish queueing from provider variance."},
    {"id": "plan-001", "role": "plan", "prompt": "Plan a safe change that adds a --provider filter to a CLI. Include validation, tests, and rollback."},
    {"id": "vision-001", "role": "vision", "prompt": "Inspect the attached image and list the three most important visible UI issues.", "requires_image": True},
    {"id": "designer-001", "role": "designer", "prompt": "Review a dense model benchmark dashboard. Propose a compact layout that supports scanning role, quality, latency, and cost."},
    {"id": "advisor-001", "role": "advisor", "prompt": "Review this policy: retry every provider error with the cheapest model. Identify reliability and observability risks and propose a safer policy."},
    {"id": "default-001", "role": "default", "prompt": "State the success criteria for a reproducible model benchmark in four bullets."},
]


class RoleBenchError(Exception):
    """User-facing validation or execution setup error."""


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def split_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_selector(value: str) -> dict[str, Any]:
    raw = value.strip().strip("'\"")
    if not raw:
        raise RoleBenchError("empty model selector")
    thinking = None
    base = raw
    if ":" in raw:
        possible_base, possible_level = raw.rsplit(":", 1)
        if possible_level in THINKING_LEVELS:
            base, thinking = possible_base, possible_level
    if "/" not in base:
        raise RoleBenchError(f"model selector must be provider/model, got: {value!r}")
    provider, model = base.split("/", 1)
    if not provider or not model:
        raise RoleBenchError(f"malformed model selector: {value!r}")
    result = {"provider": provider, "model": model, "selector": f"{provider}/{model}"}
    if thinking:
        result["thinking"] = thinking
    return result


def load_json_document(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RoleBenchError(f"cannot read JSON file {path}: {exc}") from exc


def parse_json_output(text: str) -> Any:
    stripped = text.strip()
    if not stripped:
        raise RoleBenchError("OMP returned empty JSON output")
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        values = []
        for line in stripped.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                values.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        if values:
            return values
    raise RoleBenchError("OMP returned malformed JSON output")


def _catalog_items(node: Any, provider_hint: str | None = None) -> Iterable[dict[str, Any]]:
    if isinstance(node, list):
        for item in node:
            yield from _catalog_items(item, provider_hint)
        return
    if not isinstance(node, dict):
        return
    provider = node.get("provider") or node.get("providerId") or provider_hint
    model = node.get("model") or node.get("modelId") or node.get("id") or node.get("name")
    if provider and model and isinstance(model, str):
        if "/" in model and not node.get("provider") and not node.get("providerId"):
            parsed = parse_selector(model)
            provider, model = parsed["provider"], parsed["model"]
        result = {"provider": str(provider), "model": str(model), "selector": f"{provider}/{model}"}
        for key in ("capabilities", "modalities", "input", "context", "pricing"):
            if key in node:
                result[key] = node[key]
        yield result
        return
    for key, value in node.items():
        if key in {"models", "data", "items", "results"}:
            yield from _catalog_items(value, provider or provider_hint)
        elif isinstance(value, (list, dict)):
            yield from _catalog_items(value, key if isinstance(value, list) else provider_hint)


def normalize_catalog(payload: Any) -> list[dict[str, Any]]:
    candidates: dict[str, dict[str, Any]] = {}
    for item in _catalog_items(payload):
        selector = item["selector"]
        candidates.setdefault(selector, item).update({k: v for k, v in item.items() if k not in {"selector", "provider", "model"}})
    if not candidates:
        raise RoleBenchError("model catalog contains no provider/model entries")
    return [candidates[key] for key in sorted(candidates)]


def discover_catalog(omp_bin: str, catalog_file: Path | None) -> list[dict[str, Any]]:
    if catalog_file:
        return normalize_catalog(load_json_document(catalog_file))
    try:
        completed = subprocess.run(
            [omp_bin, "models", "--json"],
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RoleBenchError(f"cannot run {omp_bin} models --json: {exc}") from exc
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise RoleBenchError(f"model discovery failed (exit {completed.returncode}): {detail}")
    return normalize_catalog(parse_json_output(completed.stdout))


def parse_model_roles(config_file: Path) -> dict[str, str]:
    try:
        lines = config_file.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise RoleBenchError(f"cannot read OMP config {config_file}: {exc}") from exc
    roles: dict[str, str] = {}
    in_roles = False
    base_indent = -1
    for line in lines:
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        indent = len(line) - len(line.lstrip(" "))
        stripped = line.strip()
        if stripped == "modelRoles:":
            in_roles, base_indent = True, indent
            continue
        if in_roles and indent <= base_indent:
            in_roles = False
        if not in_roles or ":" not in stripped:
            continue
        role, value = stripped.split(":", 1)
        role, value = role.strip(), value.strip()
        if not role or not value or value.startswith("#"):
            continue
        value = value.split(" #", 1)[0].strip().strip("'\"")
        roles[role] = value
    if not roles:
        raise RoleBenchError(f"no modelRoles found in {config_file}")
    return roles


def resolve_roles(value: str, config_file: Path) -> tuple[list[str], dict[str, str]]:
    baseline = parse_model_roles(config_file) if value == "all" else {}
    roles = list(baseline) if value == "all" else split_csv(value)
    if not roles:
        raise RoleBenchError("at least one role is required")
    if len(set(roles)) != len(roles):
        raise RoleBenchError("duplicate role selector")
    if value != "all":
        known = parse_model_roles(config_file) if config_file.exists() else {}
        unknown = [role for role in roles if known and role not in known]
        if unknown:
            raise RoleBenchError(f"unknown role(s): {', '.join(unknown)}")
        baseline = {role: known.get(role, "") for role in roles}
    return roles, baseline


def resolve_candidates(args: argparse.Namespace) -> tuple[list[dict[str, Any]], Any]:
    explicit = split_csv(args.models) + [item for value in (args.model or []) for item in split_csv(value)]
    providers = split_csv(args.providers)
    if explicit and providers and providers != ["all"]:
        raise RoleBenchError("use either explicit models or --providers, not both")
    if args.mode == "explicit" and args.assignment_file and not explicit:
        raw = parse_assignment_file(Path(args.assignment_file))
        seen: dict[str, dict[str, Any]] = {}
        for selector in raw.values():
            candidate = parse_selector(selector)
            seen[candidate["selector"]] = candidate
        return [seen[key] for key in sorted(seen)], {"source": "assignment", "file": args.assignment_file}
    if explicit:
        seen: dict[str, dict[str, Any]] = {}
        for selector in explicit:
            candidate = parse_selector(selector)
            seen[candidate["selector"]] = candidate
        return [seen[key] for key in sorted(seen)], {"source": "explicit", "selectors": explicit}
    catalog = discover_catalog(args.omp_bin, Path(args.catalog_file) if args.catalog_file else None)
    if providers and providers != ["all"]:
        catalog = [item for item in catalog if item["provider"] in set(providers)]
        if not catalog:
            raise RoleBenchError(f"no discovered models matched provider(s): {', '.join(providers)}")
    return catalog, {"source": "catalog", "providers": providers or ["all"]}


def load_tasks(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.tasks_file:
        path = Path(args.tasks_file)
        if path.suffix.lower() == ".jsonl":
            tasks = []
            try:
                for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                    if line.strip():
                        tasks.append(json.loads(line))
            except (OSError, json.JSONDecodeError) as exc:
                raise RoleBenchError(f"cannot read task JSONL {path}: {exc}") from exc
        else:
            payload = load_json_document(path)
            tasks = payload if isinstance(payload, list) else payload.get("tasks", []) if isinstance(payload, dict) else []
    else:
        tasks = CORE_TASKS if args.suite == "core" else []
    if not tasks:
        raise RoleBenchError("task suite is empty; provide --tasks-file or use --suite core")
    normalized = []
    seen = set()
    for task in tasks:
        if not isinstance(task, dict) or not task.get("id") or not task.get("role") or not task.get("prompt"):
            raise RoleBenchError("each task must contain id, role, and prompt")
        if task["id"] in seen:
            raise RoleBenchError(f"duplicate task id: {task['id']}")
        seen.add(task["id"])
        normalized.append(dict(task))
    return normalized


def parse_assignment_file(path: Path) -> dict[str, str]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise RoleBenchError(f"cannot read assignment file {path}: {exc}") from exc
    result: dict[str, str] = {}
    in_assignment = False
    base_indent = -1
    for line in lines:
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        indent = len(line) - len(line.lstrip(" "))
        stripped = line.strip()
        if stripped == "assignment:":
            in_assignment, base_indent = True, indent
            continue
        if in_assignment and indent <= base_indent:
            in_assignment = False
        if not in_assignment or ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        value = value.split(" #", 1)[0].strip().strip("'\"")
        if key.strip() and value:
            result[key.strip()] = value
    if not result:
        raise RoleBenchError(f"no assignment entries found in {path}")
    return result


def build_assignments(mode: str, roles: list[str], baseline: dict[str, str], candidates: list[dict[str, Any]], assignment_file: str | None) -> list[dict[str, Any]]:
    if mode == "explicit":
        if not assignment_file:
            raise RoleBenchError("--assignment-file is required for --mode explicit")
        raw = parse_assignment_file(Path(assignment_file))
        missing = [role for role in roles if role not in raw]
        extra = [role for role in raw if role not in roles]
        if missing or extra:
            detail = []
            if missing:
                detail.append("missing=" + ",".join(missing))
            if extra:
                detail.append("unknown=" + ",".join(extra))
            raise RoleBenchError("assignment roles do not match selection (" + "; ".join(detail) + ")")
        parsed = {role: parse_selector(raw[role]) for role in roles}
        return [{
            "id": "assignment-001",
            "mode": mode,
            "role": None,
            "candidate": None,
            "mapping": {role: item["selector"] for role, item in parsed.items()},
            "thinking": {role: item.get("thinking") for role, item in parsed.items()},
        }]
    assignments = []
    if mode == "all-same":
        for index, candidate in enumerate(candidates, 1):
            assignments.append({
                "id": f"assignment-{index:03d}",
                "mode": mode,
                "role": None,
                "candidate": candidate["selector"],
                "mapping": {role: candidate["selector"] for role in roles},
                "thinking": {role: candidate.get("thinking") for role in roles},
            })
        return assignments
    for index, candidate in enumerate(candidates, 1):
        for role_index, role in enumerate(roles, 1):
            mapping = {name: baseline.get(name, "") for name in roles}
            mapping[role] = candidate["selector"]
            assignments.append({
                "id": f"assignment-{index:03d}-{role_index:02d}",
                "mode": mode,
                "role": role,
                "candidate": candidate["selector"],
                "mapping": mapping,
                "thinking": {name: (candidate.get("thinking") if name == role else None) for name in roles},
            })
    return assignments


def task_for_assignment(task: dict[str, Any], assignment: dict[str, Any]) -> bool:
    if assignment["mode"] == "role-isolated":
        return task["role"] == assignment["role"]
    return task["role"] in assignment["mapping"]


def validate_task_coverage(tasks: list[dict[str, Any]], roles: list[str]) -> None:
    present = {task["role"] for task in tasks}
    missing = [role for role in roles if role not in present]
    if missing:
        raise RoleBenchError("no task fixture for selected role(s): " + ", ".join(missing))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def extract_output(payload: Any) -> str:
    if isinstance(payload, str):
        return payload
    if isinstance(payload, dict):
        for key in ("text", "output", "content", "message"):
            value = payload.get(key)
            if isinstance(value, str):
                return value
            if isinstance(value, (dict, list)):
                found = extract_output(value)
                if found:
                    return found
        for key in ("result", "data", "response", "events"):
            if key in payload:
                found = extract_output(payload[key])
                if found:
                    return found
    if isinstance(payload, list):
        chunks = [extract_output(item) for item in payload]
        return "\n".join(chunk for chunk in chunks if chunk)
    return ""


def extract_usage(payload: Any) -> dict[str, Any] | None:
    if isinstance(payload, dict):
        for key in ("usage", "cost"):
            value = payload.get(key)
            if isinstance(value, dict):
                return value
        for value in payload.values():
            found = extract_usage(value)
            if found:
                return found
    if isinstance(payload, list):
        for value in payload:
            found = extract_usage(value)
            if found:
                return found
    return None


def extract_cost(usage: Any) -> float | None:
    if isinstance(usage, dict):
        nested = usage.get("cost")
        if nested is not None:
            found = extract_cost(nested)
            if found is not None:
                return found
        for key in ("total", "totalCost", "total_cost", "usd", "amount"):
            value = usage.get(key)
            if isinstance(value, (int, float)):
                if key == "total" and any(token_key in usage for token_key in ("input", "output", "inputTokens", "outputTokens", "totalTokens", "reasoningTokens")):
                    continue
                return float(value)
    if isinstance(usage, (int, float)):
        return float(usage)
    return None


def load_pricing(path: str | None) -> dict[str, dict[str, float]]:
    if not path:
        return {}
    payload = load_json_document(Path(path).expanduser())
    if not isinstance(payload, dict):
        raise RoleBenchError("pricing file must be a JSON object keyed by provider/model")
    result: dict[str, dict[str, float]] = {}
    for selector, value in payload.items():
        if not isinstance(selector, str) or not isinstance(value, dict):
            raise RoleBenchError("pricing entries must map provider/model to an object")
        rates = {}
        for key in ("input_per_million", "output_per_million"):
            if key in value:
                if not isinstance(value[key], (int, float)) or value[key] < 0:
                    raise RoleBenchError(f"invalid {key} for pricing entry {selector}")
                rates[key] = float(value[key])
        result[parse_selector(selector)["selector"]] = rates
    return result


def estimate_benchmark(roles: list[str], assignments: list[dict[str, Any]], tasks: list[dict[str, Any]], runs: int, pricing: dict[str, dict[str, float]]) -> dict[str, Any]:
    rows = []
    total_calls = 0
    input_tokens = 0
    output_tokens = 0
    known_cost = 0.0
    cost_known = True
    for assignment in assignments:
        applicable = [task for task in tasks if task_for_assignment(task, assignment)]
        for task in applicable:
            model = assignment["mapping"].get(task["role"]) or assignment.get("candidate") or ""
            calls = runs
            estimated_input = max(1, math.ceil(len(task["prompt"]) / 4)) + 100
            estimated_output = int(task.get("max_output_tokens", 512))
            total_calls += calls
            input_tokens += estimated_input * calls
            output_tokens += estimated_output * calls
            rates = pricing.get(parse_selector(model)["selector"] if "/" in model else model, {}) if model else {}
            if "input_per_million" in rates and "output_per_million" in rates:
                known_cost += calls * (estimated_input * rates["input_per_million"] + estimated_output * rates["output_per_million"]) / 1_000_000
            else:
                cost_known = False
            rows.append({
                "assignment_id": assignment["id"],
                "role": task["role"],
                "model": model,
                "calls": calls,
                "input_tokens": estimated_input * calls,
                "output_tokens": estimated_output * calls,
            })
    return {
        "assignments": len(assignments),
        "calls": total_calls,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": round(known_cost, 8) if cost_known else None,
        "cost_status": "estimated" if cost_known else "unknown",
        "by_assignment": rows,
    }


def candidate_pricing(candidates: list[dict[str, Any]], supplied: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    pricing = dict(supplied)
    for candidate in candidates:
        selector = candidate["selector"]
        value = candidate.get("pricing")
        if selector in pricing or not isinstance(value, dict):
            continue
        rates = {}
        for key in ("input_per_million", "output_per_million"):
            if isinstance(value.get(key), (int, float)) and value[key] >= 0:
                rates[key] = float(value[key])
        if rates:
            pricing[selector] = rates
    return pricing


def format_plan(manifest: dict[str, Any]) -> str:
    estimate = manifest["estimate"]
    cap = manifest.get("max_cost_usd")
    cost = f"${estimate['cost_usd']:.6f}" if estimate.get("cost_usd") is not None else "unknown"
    cap_text = f"${cap:.6f}" if isinstance(cap, (int, float)) else "未设置"
    lines = [
        "Benchmark plan",
        f"- models: {', '.join(item['selector'] for item in manifest['candidates'])}",
        f"- roles: {', '.join(manifest['roles'])}",
        f"- mode/executor: {manifest['mode']} / {manifest['executor']}",
        f"- task fixtures: {len(manifest['tasks'])}",
        f"- assignments: {estimate['assignments']}",
        f"- requests: {estimate['calls']} ({manifest['runs']} run(s) per task)",
        f"- estimated tokens: input {estimate['input_tokens']}, output {estimate['output_tokens']}",
        f"- estimated cost: {cost} ({estimate['cost_status']})",
        f"- cost cap: {cap_text}",
    ]
    return "\n".join(lines)


def percentile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(len(ordered) * probability) - 1))
    return ordered[index]


def run_one(omp_bin: str, assignment: dict[str, Any], task: dict[str, Any], replicate: int, output_dir: Path, timeout: int, config_file: Path, executor: str, controller_model: str | None) -> dict[str, Any]:
    started = time.perf_counter()
    result: dict[str, Any] = {
        "timestamp": now_iso(),
        "assignment_id": assignment["id"],
        "mode": assignment["mode"],
        "role": task["role"],
        "task_id": task["id"],
        "replicate": replicate,
        "model": assignment["mapping"].get(task["role"], ""),
        "executor": executor,
        "status": "error",
    }
    if task.get("requires_image") and not task.get("image"):
        result.update({"error_type": "missing_image_fixture", "error": "task requires an image fixture"})
        result["latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
        return result
    temp_root = Path(tempfile.mkdtemp(prefix="omp-rolebench-"))
    try:
        workspace = temp_root / "workspace"
        source_workspace = task.get("workspace")
        if source_workspace:
            source = Path(source_workspace).expanduser().resolve()
            if not source.is_dir():
                raise RoleBenchError(f"task workspace does not exist: {source}")
            shutil.copytree(source, workspace)
        else:
            workspace.mkdir()
        selected_model = assignment["mapping"].get(task["role"]) or assignment.get("candidate")
        if not selected_model:
            raise RoleBenchError(f"no model selected for role {task['role']}")
        image_args: list[str] = []
        image_values = task.get("image", [])
        if isinstance(image_values, str):
            image_values = [image_values]
        for image_value in image_values:
            image_source = Path(image_value).expanduser().resolve()
            if not image_source.is_file():
                raise RoleBenchError(f"task image does not exist: {image_source}")
            image_target = workspace / image_source.name
            shutil.copy2(image_source, image_target)
            image_args.append("@" + str(image_target))
        if image_args:
            result["images"] = [item[1:] for item in image_args]
        if executor == "delegated":
            agent_name = f"bench-{assignment['id'].replace('_', '-')}-{task['role']}-{replicate}"
            result["agent"] = agent_name
            result["controller_model"] = controller_model or assignment.get("mapping", {}).get("default") or selected_model
            agent_dir = workspace / ".omp" / "agents"
            agent_dir.mkdir(parents=True, exist_ok=True)
            thinking = assignment.get("thinking", {}).get(task["role"])
            agent_lines = [
                "---",
                f"name: {agent_name}",
                "description: OMP RoleBench temporary probe agent",
                "model:",
                f"  - \"{selected_model}\"",
            ]
            if thinking:
                agent_lines.append(f"thinkingLevel: {thinking}")
            agent_lines.extend(["---", "", "Worker agent: complete only the benchmark assignment.", ""])
            (agent_dir / f"{agent_name}.md").write_text("\n".join(agent_lines), encoding="utf-8")
        overlay = temp_root / "config.yml"
        lines = ["modelRoles:"]
        for role, model in assignment["mapping"].items():
            if model:
                lines.append(f"  {role}: {model}")
        overlay.write_text("\n".join(lines) + "\n", encoding="utf-8")
        system_prompt = temp_root / "role-prompt.txt"
        system_prompt.write_text(ROLE_PROMPTS.get(task["role"], ROLE_PROMPTS["default"]) + "\n", encoding="utf-8")
        thinking = assignment.get("thinking", {}).get(task["role"])
        controller = controller_model or assignment.get("mapping", {}).get("default") or selected_model
        if executor == "delegated":
            agent_name = f"bench-{assignment['id'].replace('_', '-')}-{task['role']}-{replicate}"
            prompt = (
                f"Use the task tool exactly once with agent={agent_name!r}. "
                "Do not solve the assignment yourself. Return the delegated result verbatim.\n\n"
                f"Assignment for role {task['role']}:\n{task['prompt']}"
            )
            if image_args:
                prompt += "\nImage fixture is available at: " + ", ".join(item[1:] for item in image_args)
            selected_model = controller
        else:
            prompt = task["prompt"]
        command = [
            omp_bin,
            "-p",
            "--mode",
            "json",
            "--no-session",
            "--no-title",
            "--model",
            selected_model,
            "--config",
            str(overlay),
            "--append-system-prompt",
            str(system_prompt),
            "--cwd",
            str(workspace),
            "--auto-approve",
        ]
        if executor == "delegated":
            command.extend(["--tools", "task"])
        elif thinking:
            command.extend(["--thinking", thinking])
        command.append(prompt)
        command.extend(image_args)
        completed = subprocess.run(command, check=False, capture_output=True, text=True, timeout=timeout)
        result["exit_code"] = completed.returncode
        result["stdout"] = completed.stdout
        result["stderr"] = completed.stderr
        result["command"] = command[:-(1 + len(image_args))]
        if completed.returncode != 0:
            result["error_type"] = "omp_exit"
            result["error"] = completed.stderr.strip() or f"OMP exited with {completed.returncode}"
        else:
            try:
                payload = parse_json_output(completed.stdout)
                result["output"] = extract_output(payload)
                usage = extract_usage(payload)
                if usage is not None:
                    result["usage"] = usage
                result["status"] = "pass" if result["output"].strip() else "error"
                if executor == "delegated" and re.search(r"status=(?:failed|error)|<task-result[^>]+status=\"(?:failed|error)", result["output"], re.I):
                    result["status"] = "error"
                    result["error_type"] = "delegated_task_failure"
                if result["status"] == "error" and "error_type" not in result:
                    result["error_type"] = "empty_output"
            except RoleBenchError as exc:
                result["error_type"] = "malformed_omp_json"
                result["error"] = str(exc)
    except subprocess.TimeoutExpired as exc:
        result["error_type"] = "timeout"
        result["error"] = f"OMP exceeded {timeout}s"
        result["stdout"] = exc.stdout or ""
        result["stderr"] = exc.stderr or ""
    except (OSError, RoleBenchError) as exc:
        result["error_type"] = "executor_setup"
        result["error"] = str(exc)
    finally:
        result["latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
        shutil.rmtree(temp_root, ignore_errors=True)
    return result


def write_results(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_command(args: argparse.Namespace) -> int:
    roles, baseline = resolve_roles(args.roles, Path(args.config_file).expanduser())
    candidates, source = resolve_candidates(args)
    tasks = load_tasks(args)
    validate_task_coverage(tasks, roles)
    assignments = build_assignments(args.mode, roles, baseline, candidates, args.assignment_file)
    supplied_pricing = load_pricing(args.pricing_file)
    pricing = candidate_pricing(candidates, supplied_pricing)
    estimate = estimate_benchmark(roles, assignments, tasks, args.runs, pricing)
    output_dir = Path(args.output).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "generated_at": now_iso(),
        "omp_bin": args.omp_bin,
        "selection": source,
        "roles": roles,
        "baseline_roles": baseline,
        "mode": args.mode,
        "runs": args.runs,
        "execute": bool(args.execute),
        "executor": args.executor,
        "controller_model": args.controller_model,
        "pricing_file": args.pricing_file,
        "max_cost_usd": args.max_cost_usd,
        "estimate": estimate,
        "consent": {
            "required": bool(args.execute),
            "granted": bool(args.execute and args.confirm),
            "method": "explicit --confirm" if args.execute and args.confirm else None,
            "granted_at": now_iso() if args.execute and args.confirm else None,
        },
        "tasks": tasks,
        "candidates": candidates,
        "assignments": assignments,
    }
    write_json(output_dir / "manifest.json", manifest)
    print(format_plan(manifest), flush=True)
    if args.max_cost_usd is not None:
        if estimate["cost_status"] != "estimated":
            summary = {"run_dir": str(output_dir), "status": "blocked", "reason": "cost_unknown", "executed": False}
            write_json(output_dir / "summary.json", summary)
            print("Execution blocked: cost is unknown; provide --pricing-file with per-million-token rates.", file=sys.stderr)
            return 2
        if estimate["cost_usd"] > args.max_cost_usd:
            summary = {"run_dir": str(output_dir), "status": "blocked", "reason": "estimated_cost_exceeds_cap", "estimated_cost_usd": estimate["cost_usd"], "max_cost_usd": args.max_cost_usd, "executed": False}
            write_json(output_dir / "summary.json", summary)
            print(f"Execution blocked: estimated cost ${estimate['cost_usd']:.6f} exceeds cap ${args.max_cost_usd:.6f}.", file=sys.stderr)
            return 2
    if args.confirm and not args.execute:
        raise RoleBenchError("--confirm requires --execute")
    if args.execute and not args.confirm:
        summary = {"run_dir": str(output_dir), "status": "awaiting_consent", "reason": "missing_confirm", "executed": False, "estimate": estimate}
        write_json(output_dir / "summary.json", summary)
        print("Execution blocked: obtain user consent, then rerun with --execute --confirm.", file=sys.stderr)
        return 3
    rows = []
    actual_cost = 0.0
    actual_cost_seen = False
    budget_status = "not-set" if args.max_cost_usd is None else "running"
    stopped_reason = None
    if args.execute:
        ordered = list(assignments)
        random.Random(args.seed).shuffle(ordered)
        stop = False
        for assignment in ordered:
            applicable = [task for task in tasks if task_for_assignment(task, assignment)]
            for replicate in range(1, args.runs + 1):
                for task in applicable:
                    rows.append(run_one(
                        args.omp_bin,
                        assignment,
                        task,
                        replicate,
                        output_dir,
                        args.timeout,
                        Path(args.config_file).expanduser(),
                        args.executor,
                        args.controller_model,
                    ))
                    row = rows[-1]
                    cost = extract_cost(row.get("usage"))
                    if cost is not None:
                        actual_cost += cost
                        actual_cost_seen = True
                    if args.max_cost_usd is not None and "exit_code" in row:
                        if cost is None:
                            budget_status = "unverifiable"
                            stopped_reason = "provider_did_not_return_cost"
                            stop = True
                            break
                        if actual_cost >= args.max_cost_usd:
                            budget_status = "reached"
                            stopped_reason = "actual_cost_reached_cap"
                            stop = True
                            break
                if stop:
                    break
            if stop:
                break
        write_results(output_dir / "results.jsonl", rows)
    summary = {
        "run_dir": str(output_dir),
        "status": "completed" if args.execute else "preview",
        "assignments": len(assignments),
        "tasks": len(tasks),
        "result_rows": len(rows),
        "executed": bool(args.execute),
        "estimated_cost_usd": estimate["cost_usd"],
        "actual_cost_usd": round(actual_cost, 8) if args.execute and actual_cost_seen else None,
        "max_cost_usd": args.max_cost_usd,
        "budget_status": budget_status,
        "stopped_reason": stopped_reason,
    }
    write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def report_command(args: argparse.Namespace) -> int:
    run_dir = Path(args.run).expanduser()
    manifest_path = run_dir / "manifest.json"
    results_path = run_dir / "results.jsonl"
    manifest = load_json_document(manifest_path)
    rows = []
    if results_path.exists():
        for number, line in enumerate(results_path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RoleBenchError(f"malformed results.jsonl line {number}: {exc}") from exc
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row.get("role", ""), row.get("model", ""))].append(row)
    lines = [
        "# OMP RoleBench Report",
        "",
        f"- Generated: {manifest.get('generated_at', 'unknown')}",
        f"- Mode: `{manifest.get('mode', 'unknown')}`",
        f"- Roles: {', '.join(manifest.get('roles', []))}",
        f"- Assignments: {len(manifest.get('assignments', []))}",
        f"- Result rows: {len(rows)}",
        "",
        "| Role | Model | Runs | Pass | Error | Pass rate | P50 latency (ms) | P95 latency (ms) | Avg output chars | Cost |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for (role, model), values in sorted(groups.items()):
        passes = sum(value.get("status") == "pass" for value in values)
        errors = len(values) - passes
        latency = [float(value["latency_ms"]) for value in values if isinstance(value.get("latency_ms"), (int, float))]
        output_size = [len(value.get("output", "")) for value in values]
        costs = [cost for value in values if (cost := extract_cost(value.get("usage"))) is not None]
        p50 = percentile(latency, 0.50)
        p95 = percentile(latency, 0.95)
        p50_text = f"{p50:.1f}" if p50 is not None else "unknown"
        p95_text = f"{p95:.1f}" if p95 is not None else "unknown"
        avg_output = f"{sum(output_size) / len(output_size):.1f}" if output_size else "0"
        rate = f"{passes / len(values):.1%}" if values else "0.0%"
        total_cost = f"${sum(costs):.6f}" if costs else "unknown"
        lines.append(f"| `{role}` | `{model}` | {len(values)} | {passes} | {errors} | {rate} | {p50_text} | {p95_text} | {avg_output} | {total_cost} |")
    if not rows:
        lines.extend(["", "No executed results found. This run contains a matrix preview only."])
    report = "\n".join(lines) + "\n"
    if args.format == "json":
        serializable_groups = {
            f"{role}/{model}": values for (role, model), values in groups.items()
        }
        print(json.dumps({"manifest": manifest, "groups": serializable_groups}, ensure_ascii=False, indent=2))
    elif args.output:
        output = Path(args.output).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(report, encoding="utf-8")
        print(output)
    else:
        print(report, end="")
    return 0


def discover_command(args: argparse.Namespace) -> int:
    catalog = discover_catalog(args.omp_bin, Path(args.catalog_file) if args.catalog_file else None)
    providers = split_csv(args.providers)
    if providers and providers != ["all"]:
        catalog = [item for item in catalog if item["provider"] in set(providers)]
    payload = {"generated_at": now_iso(), "count": len(catalog), "models": catalog}
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        output = Path(args.output).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


def add_selection_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--models", help="comma-separated explicit provider/model selectors")
    parser.add_argument("--model", action="append", help="explicit provider/model selector; repeatable")
    parser.add_argument("--providers", default="all", help="all or comma-separated provider names")
    parser.add_argument("--catalog-file", help="saved omp models --json response")
    parser.add_argument("--omp-bin", default="omp", help="OMP executable")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark OMP models across modelRoles")
    subparsers = parser.add_subparsers(dest="command", required=True)
    discover = subparsers.add_parser("discover", help="discover and normalize OMP models")
    add_selection_args(discover)
    discover.add_argument("--output", help="also save normalized catalog JSON")
    run = subparsers.add_parser("run", help="generate or execute a benchmark matrix")
    add_selection_args(run)
    run.add_argument("--roles", default="all", help="all or comma-separated role names")
    run.add_argument("--config-file", default="~/.omp/agent/config.yml", help="OMP config containing modelRoles")
    run.add_argument("--mode", choices=("role-isolated", "all-same", "explicit"), default="role-isolated")
    run.add_argument("--assignment-file", help="YAML-like role assignment file for explicit mode")
    run.add_argument("--suite", choices=("core", "none"), default="core")
    run.add_argument("--tasks-file", help="JSON or JSONL task fixtures")
    run.add_argument("--runs", type=int, default=1)
    run.add_argument("--execute", action="store_true", help="invoke OMP; otherwise only write manifest")
    run.add_argument("--executor", choices=("direct", "delegated"), default="direct", help="direct model call or a temporary OMP task agent")
    run.add_argument("--controller-model", help="controller model used by delegated executor")
    run.add_argument("--confirm", action="store_true", help="confirm the printed plan and cost estimate before execution")
    run.add_argument("--pricing-file", help="JSON rates keyed by provider/model, in USD per million input/output tokens")
    run.add_argument("--max-cost-usd", "--budget-usd", dest="max_cost_usd", type=float, help="maximum allowed estimated/observed benchmark cost in USD")
    run.add_argument("--timeout", type=int, default=300)
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--output", required=True)
    report = subparsers.add_parser("report", help="summarize an existing run")
    report.add_argument("--run", required=True)
    report.add_argument("--format", choices=("markdown", "json"), default="markdown")
    report.add_argument("--output")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "discover":
            return discover_command(args)
        if args.command == "run":
            if args.runs < 1:
                raise RoleBenchError("--runs must be >= 1")
            if args.timeout < 1:
                raise RoleBenchError("--timeout must be >= 1")
            if args.max_cost_usd is not None and args.max_cost_usd < 0:
                raise RoleBenchError("--max-cost-usd must be >= 0")
            return run_command(args)
        if args.command == "report":
            return report_command(args)
        raise RoleBenchError(f"unknown command: {args.command}")
    except RoleBenchError as exc:
        print(f"rolebench: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
