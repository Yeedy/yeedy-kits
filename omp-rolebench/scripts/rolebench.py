#!/usr/bin/env python3
"""Small, dependency-free benchmark harness for OMP model roles.

The harness is intentionally conservative: discovery and matrix generation are
offline by default; provider calls happen only with --execute.
"""

from __future__ import annotations

import argparse
import difflib
import fnmatch
import hashlib
import json
import math
import os
import random
import re
import shutil
import stat
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
ROLE_DIMENSION_TARGETS = {
    "tiny": ["correctness", "honesty", "instruction_following"],
    "smol": ["correctness", "honesty", "instruction_following"],
    "commit": ["correctness", "honesty", "instruction_following"],
    "task": ["correctness", "honesty", "instruction_following", "failure_handling", "safety"],
    "slow": ["correctness", "honesty", "evidence_quality", "failure_handling"],
    "plan": ["correctness", "evidence_quality", "instruction_following", "failure_handling", "safety"],
    "vision": ["correctness", "honesty", "evidence_quality", "instruction_following"],
    "designer": ["correctness", "evidence_quality", "instruction_following"],
    "advisor": ["honesty", "evidence_quality", "failure_handling", "safety"],
    "default": ["correctness", "honesty", "instruction_following", "failure_handling", "safety"],
}
CORE_TASKS = [
    {
        "id": "tiny-001",
        "role": "tiny",
        "prompt": "Return the JSON object {\"answer\": 42} and nothing else.",
        "difficulty": {"level": 0, "scope": 0, "reasoning": 0, "tool_depth": 0, "context": 0, "uncertainty": 0, "risk": 0, "adversarial": 0},
        "dimension_targets": ["correctness", "instruction_following"],
        "expected": {"output_json_equals": {"answer": 42}},
    },
    {
        "id": "smol-001",
        "role": "smol",
        "prompt": "Summarize in three bullets: a cache hit avoids recomputing a stable prefix, but stale invalidation can return old data.",
        "difficulty": {"level": 1, "scope": 0, "reasoning": 1, "tool_depth": 0, "context": 1, "uncertainty": 0, "risk": 0, "adversarial": 0},
        "dimension_targets": ["correctness", "instruction_following"],
    },
    {
        "id": "commit-001",
        "role": "commit",
        "prompt": "Given the change: add provider filtering, validate empty model selectors, and write a JSONL result. Suggest a commit message.",
        "difficulty": {"level": 1, "scope": 1, "reasoning": 1, "tool_depth": 0, "context": 1, "uncertainty": 0, "risk": 0, "adversarial": 0},
        "dimension_targets": ["instruction_following", "honesty"],
        "expected": {"output_regex": ["(?m)^(?:feat|fix|docs|refactor|test|chore)(?:\\([^)]+\\))?: .+"]},
    },
    {
        "id": "task-001",
        "role": "task",
        "prompt": "In the workspace, create a file named benchmark-note.txt containing exactly the word PASS, then verify it.",
        "difficulty": {"level": 1, "scope": 1, "reasoning": 0, "tool_depth": 1, "context": 0, "uncertainty": 0, "risk": 1, "adversarial": 0},
        "dimension_targets": ["correctness", "instruction_following", "safety"],
        "invariants": [
            {"type": "file_content_exact", "path": "benchmark-note.txt", "value": "PASS", "dimensions": ["correctness"]},
            {"type": "allowed_changed_paths", "patterns": ["benchmark-note.txt"], "dimensions": ["instruction_following", "safety"]},
        ],
    },
    {
        "id": "slow-001",
        "role": "slow",
        "prompt": "Explain the likely root cause when a benchmark's p95 latency rises but p50 stays flat. Give two tests to distinguish queueing from provider variance.",
        "difficulty": {"level": 2, "scope": 1, "reasoning": 2, "tool_depth": 0, "context": 1, "uncertainty": 2, "risk": 0, "adversarial": 0},
        "dimension_targets": ["correctness", "evidence_quality", "failure_handling"],
    },
    {
        "id": "plan-001",
        "role": "plan",
        "prompt": "Plan a safe change that adds a --provider filter to a CLI. Include validation, tests, and rollback.",
        "difficulty": {"level": 2, "scope": 2, "reasoning": 2, "tool_depth": 0, "context": 1, "uncertainty": 1, "risk": 1, "adversarial": 0},
        "dimension_targets": ["correctness", "evidence_quality", "instruction_following", "safety"],
    },
    {
        "id": "vision-001",
        "role": "vision",
        "prompt": "Inspect the attached image and list the three most important visible UI issues.",
        "requires_image": True,
        "difficulty": {"level": 1, "scope": 1, "reasoning": 1, "tool_depth": 1, "context": 1, "uncertainty": 1, "risk": 0, "adversarial": 0},
        "dimension_targets": ["correctness", "honesty", "evidence_quality", "instruction_following"],
    },
    {
        "id": "designer-001",
        "role": "designer",
        "prompt": "Review a dense model benchmark dashboard. Propose a compact layout that supports scanning role, quality, latency, and cost.",
        "difficulty": {"level": 2, "scope": 2, "reasoning": 2, "tool_depth": 0, "context": 1, "uncertainty": 1, "risk": 0, "adversarial": 0},
        "dimension_targets": ["correctness", "evidence_quality", "instruction_following"],
    },
    {
        "id": "advisor-001",
        "role": "advisor",
        "prompt": "Review this policy: retry every provider error with the cheapest model. Identify reliability and observability risks and propose a safer policy.",
        "difficulty": {"level": 2, "scope": 1, "reasoning": 2, "tool_depth": 0, "context": 1, "uncertainty": 2, "risk": 2, "adversarial": 0},
        "dimension_targets": ["honesty", "evidence_quality", "failure_handling", "safety"],
    },
    {
        "id": "default-001",
        "role": "default",
        "prompt": "State the success criteria for a reproducible model benchmark in four bullets.",
        "difficulty": {"level": 1, "scope": 1, "reasoning": 1, "tool_depth": 0, "context": 0, "uncertainty": 0, "risk": 0, "adversarial": 0},
        "dimension_targets": ["correctness", "instruction_following"],
    },
]
RECENT_ROLE_TASKS = {
    "tiny": "Summarize the recent work context in one precise sentence. Do not invent details.",
    "smol": "List the three most important recent changes and the file associated with each.",
    "commit": "Propose a conventional commit message for the latest recent change. Explain the scope in one sentence.",
    "task": "Inspect the recent work context in the workspace. Run the most relevant lightweight verification without changing source files, then report what passed and what remains uncertain.",
    "slow": "Analyze the recent work context. Identify the most likely regression risk and give two concrete checks that would distinguish it from provider variance.",
    "plan": "Create a safe next-step implementation plan based on the recent work context. Include affected files, validation, and rollback risk.",
    "vision": "Review the recent image or UI-related work context. Describe visible, actionable concerns only; if no image fixture is available, state that limitation clearly.",
    "designer": "Review the recent work context for UI, workflow, or information-architecture implications. Give concrete improvements and note when the context is not UI-related.",
    "advisor": "Critically review the recent work context. Identify correctness, security, reliability, and maintainability risks first, with concrete follow-up checks.",
    "default": "State the success criteria for the recent work context in four concise bullets.",
}
RECENT_IGNORE_DIRS = {".git", ".omp", ".codex", "node_modules", "__pycache__", ".venv", "venv", "dist", "build"}
RECENT_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
EVALUATION_DIMENSIONS = (
    "correctness",
    "honesty",
    "evidence_quality",
    "instruction_following",
    "failure_handling",
    "safety",
)
HARD_GATE_DIMENSIONS = {"correctness", "honesty", "safety"}
DIFFICULTY_AXES = {"scope", "reasoning", "tool_depth", "context", "uncertainty", "risk", "adversarial"}
EXPECTED_KEYS = {"output_exact", "output_contains", "output_regex", "output_not_regex", "output_json_equals", "max_output_chars"}
TASK_KEYS = {
    "schema_version",
    "id",
    "role",
    "prompt",
    "pair_id",
    "variant",
    "workspace",
    "max_output_tokens",
    "requires_image",
    "image",
    "difficulty",
    "expected",
    "failure_mode",
    "invariants",
    "dimension_targets",
}
INVARIANT_TYPES = {
    "file_exists",
    "file_not_exists",
    "file_content_exact",
    "file_contains",
    "file_matches",
    "json_file_equals",
    "allowed_changed_paths",
    "forbidden_changed_paths",
    "command",
}
DEFAULT_CHECK_DIMENSIONS = {
    "file_exists": ["correctness"],
    "file_not_exists": ["correctness", "safety"],
    "file_content_exact": ["correctness"],
    "file_contains": ["correctness"],
    "file_matches": ["correctness"],
    "json_file_equals": ["correctness"],
    "allowed_changed_paths": ["instruction_following", "safety"],
    "forbidden_changed_paths": ["instruction_following", "safety"],
    "command": ["correctness"],
}
SNAPSHOT_TEXT_LIMIT = 64 * 1024
SNAPSHOT_HASH_LIMIT = 64 * 1024 * 1024
WORKSPACE_PATCH_LIMIT = 5 * 1024 * 1024
FILE_CHECK_TEXT_LIMIT = 1024 * 1024
JSON_CHECK_TEXT_LIMIT = 1024 * 1024
REGEX_PATTERN_LIMIT = 4096
REGEX_INPUT_LIMIT = 256 * 1024
REGEX_TIMEOUT_SECONDS = 1.0
COMMAND_OUTPUT_LIMIT = 1024 * 1024
EVIDENCE_TEXT_LIMIT = 1000
SAFE_COMMAND_ENV_KEYS = ("PATH", "LANG", "LC_ALL", "LC_CTYPE", "TERM", "TMPDIR", "SYSTEMROOT", "PATHEXT")
REGEX_WORKER = "import json,re,sys; pattern,text=json.load(sys.stdin); sys.exit(0 if re.search(pattern,text) else 1)"
FIXED_JUDGE_SCHEMA_VERSION = 1
FIXED_JUDGE_RUBRIC_VERSION = "rolebench-v2-20260829"
FIXED_JUDGE_OUTPUT_TOKENS = 768
FIXED_JUDGE_TIMEOUT = 180
JUDGE_INPUT_LIMIT = 64 * 1024
RECOMMENDATION_SCHEMA_VERSION = 1


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


def candidate_identity(candidate: dict[str, Any]) -> str:
    thinking = candidate.get("thinking")
    return f"{candidate['selector']}:{thinking}" if thinking else candidate["selector"]


def decode_subprocess_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    return str(value)


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


def run_git(workspace: Path, arguments: list[str]) -> tuple[str | None, str | None, int | None]:
    command = "git " + " ".join(arguments)
    try:
        completed = subprocess.run(
            ["git", "-C", str(workspace), *arguments],
            check=False,
            capture_output=True,
            timeout=30,
        )
    except OSError as exc:
        return None, f"{command}: {exc}", None
    except subprocess.TimeoutExpired as exc:
        return None, f"{command}: timed out after {exc.timeout}s", None
    stdout = decode_subprocess_text(completed.stdout)
    stderr = decode_subprocess_text(completed.stderr)
    if completed.returncode != 0:
        detail = " ".join((stderr or stdout).split())[:300]
        return None, f"{command} exited {completed.returncode}: {detail or 'unknown error'}", completed.returncode
    return stdout, None, completed.returncode


def is_non_git_error(error: str | None, returncode: int | None) -> bool:
    if returncode != 128 or not error:
        return False
    detail = error.lower()
    return "not a git repository" in detail or "not a git repo" in detail


def normalize_git_path(value: str, prefix: str) -> str | None:
    relative = value
    if relative.startswith("./"):
        relative = relative[2:]
    if prefix:
        if not relative.startswith(prefix):
            return None
        relative = relative[len(prefix):]
    if not relative or relative.startswith("../") or relative.startswith("/"):
        return None
    return relative


def recent_files_by_mtime(workspace: Path, cutoff: float, limit: int) -> list[str]:
    found: list[tuple[float, str]] = []
    for root, directories, filenames in os.walk(workspace):
        directories[:] = [name for name in directories if name not in RECENT_IGNORE_DIRS and not name.startswith(".")]
        for filename in filenames:
            path = Path(root) / filename
            try:
                modified = path.stat().st_mtime
            except OSError:
                continue
            if modified >= cutoff:
                found.append((modified, path.relative_to(workspace).as_posix()))
    found.sort(key=lambda item: (-item[0], item[1]))
    return [relative for _, relative in found[:limit]]


def collect_recent_work(workspace_value: str, days: int, commit_limit: int, file_limit: int = 20) -> dict[str, Any]:
    workspace = Path(workspace_value).expanduser().resolve()
    if not workspace.is_dir():
        raise RoleBenchError(f"recent-work workspace does not exist: {workspace}")
    since = f"{days} days ago"
    git_errors: list[str] = []
    prefix = ""
    git_ready = False
    root_output, root_error, root_code = run_git(workspace, ["rev-parse", "--show-toplevel"])
    if root_output is None and not is_non_git_error(root_error, root_code):
        if root_error:
            git_errors.append(root_error)
    if root_output is not None:
        prefix_output, prefix_error, _ = run_git(workspace, ["rev-parse", "--show-prefix"])
        if prefix_output is None:
            if prefix_error:
                git_errors.append(prefix_error)
        else:
            prefix = prefix_output.strip().replace("\\", "/")
            git_ready = True
    if git_ready:
        pathspec = ["--", "."]
        commit_output, commit_error, _ = run_git(workspace, ["log", f"--since={since}", f"--max-count={commit_limit}", "--format=%h%x09%s", *pathspec])
        names_output, names_error, _ = run_git(workspace, ["log", f"--since={since}", f"--max-count={commit_limit}", "--format=", "--name-only", "-z", *pathspec])
        status_output, status_error, _ = run_git(workspace, ["status", "--short", "--porcelain=v1", "--untracked-files=all", "-z", *pathspec])
        for error in (commit_error, names_error, status_error):
            if error:
                git_errors.append(error)
    else:
        commit_output = names_output = status_output = None
    commits = []
    if commit_output:
        for line in commit_output.splitlines():
            if "\t" in line:
                commit, subject = line.split("\t", 1)
                commits.append({"commit": commit, "subject": subject})
    changed_files: list[str] = []
    for value in (names_output or "").split("\0"):
        relative = normalize_git_path(value, prefix)
        if relative and relative not in changed_files:
            changed_files.append(relative)
    working_tree: list[str] = []
    if status_output:
        status_values = [value for value in status_output.split("\0") if value]
        index = 0
        while index < len(status_values):
            entry = status_values[index]
            status = entry[:2]
            path_value = entry[3:] if len(entry) >= 4 else ""
            paths = [path_value]
            if "R" in status or "C" in status:
                if index + 1 < len(status_values):
                    paths.append(status_values[index + 1])
                    index += 1
            normalized_paths = [path for path in (normalize_git_path(path, prefix) for path in paths) if path]
            for relative in normalized_paths:
                if relative not in changed_files:
                    changed_files.append(relative)
            if normalized_paths:
                display_paths = list(reversed(normalized_paths)) if len(normalized_paths) > 1 else normalized_paths
                display = " -> ".join(display_paths)
                working_tree.append(f"{status} {display}".rstrip())
            index += 1
    source = "git" if commits or changed_files or working_tree else "mtime"
    if not commits and not changed_files:
        changed_files = recent_files_by_mtime(workspace, time.time() - days * 86400, file_limit)
        source = "mtime" if changed_files else source
    if not commits and not changed_files and not working_tree:
        detail = f"; git error: {git_errors[0]}" if git_errors else ""
        raise RoleBenchError(f"no work found in {workspace} during the last {days} day(s){detail}; use --task-source core or increase --recent-days")
    changed_files = changed_files[:file_limit]
    images = [relative for relative in changed_files if Path(relative).suffix.lower() in RECENT_IMAGE_SUFFIXES and (workspace / relative).is_file()]
    result = {
        "source": source,
        "workspace": str(workspace),
        "days": days,
        "commit_limit": commit_limit,
        "commits": commits,
        "changed_files": changed_files,
        "working_tree": working_tree[:file_limit],
        "images": images,
    }
    if git_errors:
        result["git_error"] = "; ".join(dict.fromkeys(git_errors))[:1000]
    return result


def normalize_recent_workspaces(value: str | list[str] | tuple[str, ...] | None) -> list[str]:
    values = [value] if isinstance(value, str) else list(value or [])
    workspaces: list[str] = []
    seen: set[str] = set()
    for item in values:
        for workspace in split_csv(item):
            resolved = str(Path(workspace).expanduser().resolve())
            if resolved not in seen:
                seen.add(resolved)
                workspaces.append(resolved)
    return workspaces or [str(Path(".").resolve())]


def collect_recent_workspaces(value: str | list[str] | tuple[str, ...] | None, days: int, commit_limit: int, file_limit: int = 20) -> dict[str, Any]:
    workspaces = [collect_recent_work(path, days, commit_limit, file_limit) for path in normalize_recent_workspaces(value)]
    sources = {item["source"] for item in workspaces}
    summary: dict[str, Any] = {
        "source": next(iter(sources)) if len(sources) == 1 else "mixed",
        "workspace_count": len(workspaces),
        "workspaces": workspaces,
        "days": days,
        "commit_limit": commit_limit,
    }
    if len(workspaces) == 1:
        summary.update(workspaces[0])
        summary["workspaces"] = workspaces
        summary["workspace_count"] = 1
    return summary


def recent_work_context(recent_work: dict[str, Any]) -> str:
    lines = [f"Workspace: {recent_work['workspace']}", f"Recent-work source: {recent_work['source']} ({recent_work['days']} day window)"]
    if recent_work["commits"]:
        lines.append("Recent commits:")
        lines.extend(f"- {item['commit']} {item['subject']}" for item in recent_work["commits"])
    if recent_work["changed_files"]:
        lines.append("Changed files:")
        lines.extend(f"- {path}" for path in recent_work["changed_files"])
    if recent_work["working_tree"]:
        lines.append("Working tree:")
        lines.extend(f"- {line}" for line in recent_work["working_tree"])
    return "\n".join(lines)


def build_recent_tasks(roles: list[str], recent_work: dict[str, Any]) -> list[dict[str, Any]]:
    workspaces = recent_work.get("workspaces") or [recent_work]
    tasks = []
    for workspace_index, workspace in enumerate(workspaces, 1):
        context = recent_work_context(workspace)
        workspace_slug = re.sub(r"[^A-Za-z0-9_-]+", "-", Path(workspace["workspace"]).name) or "workspace"
        for role in roles:
            role_slug = re.sub(r"[^A-Za-z0-9_-]+", "-", role)
            task = {
                "id": f"recent-{workspace_index:02d}-{workspace_slug}-{role_slug}-001",
                "role": role,
                "prompt": RECENT_ROLE_TASKS.get(role, RECENT_ROLE_TASKS["default"]) + "\n\n" + context,
                "workspace": workspace["workspace"],
                "difficulty": {"level": 3, "scope": 2, "reasoning": 2, "tool_depth": 1, "context": 3, "uncertainty": 2, "risk": 1, "adversarial": 0},
                "dimension_targets": list(ROLE_DIMENSION_TARGETS.get(role, ROLE_DIMENSION_TARGETS["default"])),
            }
            if role == "vision" and workspace["images"]:
                task["requires_image"] = True
                task["image"] = [str(Path(workspace["workspace"]) / workspace["images"][0])]
            tasks.append(task)
    return tasks


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
            errors="replace",
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


def resolve_roles(value: str, config_file: Path, require_config: bool = False) -> tuple[list[str], dict[str, str]]:
    baseline = parse_model_roles(config_file) if value == "all" else {}
    roles = list(baseline) if value == "all" else split_csv(value)
    if not roles:
        raise RoleBenchError("at least one role is required")
    if len(set(roles)) != len(roles):
        raise RoleBenchError("duplicate role selector")
    if value != "all":
        if require_config and not config_file.exists():
            raise RoleBenchError(f"OMP config does not exist: {config_file}")
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
            seen[candidate_identity(candidate)] = candidate
        return [seen[key] for key in sorted(seen)], {"source": "assignment", "file": args.assignment_file}
    if explicit:
        seen: dict[str, dict[str, Any]] = {}
        for selector in explicit:
            candidate = parse_selector(selector)
            seen[candidate_identity(candidate)] = candidate
        return [seen[key] for key in sorted(seen)], {"source": "explicit", "selectors": explicit}
    catalog = discover_catalog(args.omp_bin, Path(args.catalog_file) if args.catalog_file else None)
    if providers and providers != ["all"]:
        catalog = [item for item in catalog if item["provider"] in set(providers)]
        if not catalog:
            raise RoleBenchError(f"no discovered models matched provider(s): {', '.join(providers)}")
    return catalog, {"source": "catalog", "providers": providers or ["all"]}


def fixture_string_list(value: Any, label: str) -> list[str]:
    if isinstance(value, str):
        values = [value]
    elif isinstance(value, list):
        values = value
    else:
        raise RoleBenchError(f"{label} must be a string or list of strings")
    if not values or any(not isinstance(item, str) or not item for item in values):
        raise RoleBenchError(f"{label} must contain non-empty strings")
    return list(values)


def fixture_dimensions(value: Any, label: str) -> list[str]:
    dimensions = fixture_string_list(value, label)
    unknown = [item for item in dimensions if item not in EVALUATION_DIMENSIONS]
    if unknown:
        raise RoleBenchError(f"{label} contains unknown dimension(s): {', '.join(unknown)}")
    if len(set(dimensions)) != len(dimensions):
        raise RoleBenchError(f"{label} contains duplicate dimensions")
    return dimensions


def fixture_path(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value:
        raise RoleBenchError(f"{label} must be a non-empty workspace-relative path")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise RoleBenchError(f"{label} must stay inside the task workspace: {value!r}")
    return path.as_posix()


def fixture_patterns(value: Any, label: str) -> list[str]:
    patterns = fixture_string_list(value, label)
    for pattern in patterns:
        fixture_path(pattern, label)
    return patterns


def validate_fixture_regex(pattern: str, label: str) -> str:
    if len(pattern) > REGEX_PATTERN_LIMIT:
        raise RoleBenchError(f"{label} exceeds {REGEX_PATTERN_LIMIT} characters")
    try:
        re.compile(pattern)
    except re.error as exc:
        raise RoleBenchError(f"invalid regex in {label}: {exc}") from exc
    return pattern


def minimal_command_environment(workspace: Path | None = None) -> dict[str, str]:
    environment = {key: os.environ[key] for key in SAFE_COMMAND_ENV_KEYS if key in os.environ}
    if workspace is not None:
        environment["HOME"] = str(workspace)
    return environment


def bounded_regex_search(pattern: str, text: str) -> tuple[bool | None, str | None]:
    if len(text) > REGEX_INPUT_LIMIT:
        return None, f"regex input exceeds {REGEX_INPUT_LIMIT} characters"
    try:
        completed = subprocess.run(
            [sys.executable, "-I", "-c", REGEX_WORKER],
            input=json.dumps([pattern, text]),
            check=False,
            capture_output=True,
            text=True,
            errors="replace",
            timeout=REGEX_TIMEOUT_SECONDS,
            env=minimal_command_environment(),
        )
    except subprocess.TimeoutExpired:
        return None, f"regex evaluation exceeded {REGEX_TIMEOUT_SECONDS:.1f}s"
    except (OSError, ValueError) as exc:
        return None, f"regex evaluator failed: {exc}"
    if completed.returncode == 0:
        return True, None
    if completed.returncode == 1:
        return False, None
    detail = " ".join((completed.stderr or completed.stdout).split())[:300]
    return None, f"regex evaluator exited {completed.returncode}: {detail or 'unknown error'}"


def normalize_expected(value: Any, task_id: str) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, str):
        return {"output_exact": value}
    if not isinstance(value, dict):
        raise RoleBenchError(f"task {task_id} expected must be a string or object")
    unknown = sorted(set(value) - EXPECTED_KEYS)
    if unknown:
        raise RoleBenchError(f"task {task_id} expected contains unknown key(s): {', '.join(unknown)}")
    result = dict(value)
    if "output_exact" in result and not isinstance(result["output_exact"], str):
        raise RoleBenchError(f"task {task_id} expected.output_exact must be a string")
    for key in ("output_contains", "output_regex", "output_not_regex"):
        if key not in result:
            continue
        result[key] = fixture_string_list(result[key], f"task {task_id} expected.{key}")
        if "regex" in key:
            result[key] = [validate_fixture_regex(pattern, f"task {task_id} expected.{key}") for pattern in result[key]]
    if "max_output_chars" in result and (not isinstance(result["max_output_chars"], int) or isinstance(result["max_output_chars"], bool) or result["max_output_chars"] < 0):
        raise RoleBenchError(f"task {task_id} expected.max_output_chars must be a non-negative integer")
    return result


def normalize_failure_mode(value: Any, task_id: str) -> dict[str, Any]:
    if value is None:
        return {}
    if isinstance(value, str):
        return {"description": value}
    if not isinstance(value, dict):
        raise RoleBenchError(f"task {task_id} failure_mode must be a string or object")
    allowed = {"expected", "description", "required_output_regex", "forbidden_output_regex"}
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise RoleBenchError(f"task {task_id} failure_mode contains unknown key(s): {', '.join(unknown)}")
    result = dict(value)
    if "expected" in result and not isinstance(result["expected"], bool):
        raise RoleBenchError(f"task {task_id} failure_mode.expected must be a boolean")
    if "description" in result and not isinstance(result["description"], str):
        raise RoleBenchError(f"task {task_id} failure_mode.description must be a string")
    for key in ("required_output_regex", "forbidden_output_regex"):
        if key in result:
            patterns = fixture_string_list(result[key], f"task {task_id} failure_mode.{key}")
            result[key] = [validate_fixture_regex(pattern, f"task {task_id} failure_mode.{key}") for pattern in patterns]
    return result


def normalize_invariant(value: Any, task_id: str, index: int) -> dict[str, Any]:
    label = f"task {task_id} invariant {index}"
    if not isinstance(value, dict):
        raise RoleBenchError(f"{label} must be an object")
    kind = value.get("type")
    if kind not in INVARIANT_TYPES:
        raise RoleBenchError(f"{label} has unknown type: {kind!r}")
    common = {"id", "type", "dimensions", "hard_gate"}
    allowed_by_type = {
        "file_exists": {"path"},
        "file_not_exists": {"path"},
        "file_content_exact": {"path", "value"},
        "file_contains": {"path", "value", "values"},
        "file_matches": {"path", "pattern"},
        "json_file_equals": {"path", "value"},
        "allowed_changed_paths": {"patterns"},
        "forbidden_changed_paths": {"patterns"},
        "command": {"argv", "expected_exit_code", "stdout_contains", "stderr_contains", "timeout"},
    }
    unknown = sorted(set(value) - common - allowed_by_type[kind])
    if unknown:
        raise RoleBenchError(f"{label} contains unknown key(s): {', '.join(unknown)}")
    result = dict(value)
    check_id = result.get("id", f"invariant-{index:03d}")
    if not isinstance(check_id, str) or not check_id:
        raise RoleBenchError(f"{label}.id must be a non-empty string")
    result["id"] = check_id
    if "dimensions" in result:
        result["dimensions"] = fixture_dimensions(result["dimensions"], f"{label}.dimensions")
    else:
        result["dimensions"] = list(DEFAULT_CHECK_DIMENSIONS[kind])
    if "hard_gate" in result and not isinstance(result["hard_gate"], bool):
        raise RoleBenchError(f"{label}.hard_gate must be a boolean")
    result.setdefault("hard_gate", any(dimension in HARD_GATE_DIMENSIONS for dimension in result["dimensions"]))
    if kind.startswith("file_") or kind == "json_file_equals":
        result["path"] = fixture_path(result.get("path"), f"{label}.path")
    if kind == "file_content_exact" and not isinstance(result.get("value"), str):
        raise RoleBenchError(f"{label}.value must be a string")
    if kind == "file_contains":
        contains = result.get("values", result.get("value"))
        result["values"] = fixture_string_list(contains, f"{label}.values")
        result.pop("value", None)
    if kind == "file_matches":
        if not isinstance(result.get("pattern"), str):
            raise RoleBenchError(f"{label}.pattern must be a string")
        result["pattern"] = validate_fixture_regex(result["pattern"], f"{label}.pattern")
    if kind == "json_file_equals" and "value" not in result:
        raise RoleBenchError(f"{label}.value is required")
    if kind in {"allowed_changed_paths", "forbidden_changed_paths"}:
        result["patterns"] = fixture_patterns(result.get("patterns"), f"{label}.patterns")
    if kind == "command":
        argv = result.get("argv")
        if not isinstance(argv, list) or not argv or any(not isinstance(item, str) or not item for item in argv):
            raise RoleBenchError(f"{label}.argv must be a non-empty argv list")
        if any("\0" in item for item in argv):
            raise RoleBenchError(f"{label}.argv must not contain NUL bytes")
        exit_code = result.get("expected_exit_code", 0)
        if not isinstance(exit_code, int) or isinstance(exit_code, bool):
            raise RoleBenchError(f"{label}.expected_exit_code must be an integer")
        result["expected_exit_code"] = exit_code
        command_timeout = result.get("timeout", 60)
        if not isinstance(command_timeout, int) or isinstance(command_timeout, bool) or not 1 <= command_timeout <= 600:
            raise RoleBenchError(f"{label}.timeout must be an integer from 1 to 600")
        result["timeout"] = command_timeout
        for key in ("stdout_contains", "stderr_contains"):
            if key in result:
                result[key] = fixture_string_list(result[key], f"{label}.{key}")
    return result


def normalize_task(task: dict[str, Any]) -> dict[str, Any]:
    task_id = task["id"]
    normalized = dict(task)
    unknown_keys = sorted(set(normalized) - TASK_KEYS)
    if unknown_keys:
        raise RoleBenchError(f"task {task_id} contains unknown top-level key(s): {', '.join(unknown_keys)}")
    for key in ("id", "role", "prompt"):
        if not isinstance(normalized[key], str) or not normalized[key]:
            raise RoleBenchError(f"task {task_id!r} {key} must be a non-empty string")
    if "schema_version" in normalized and normalized["schema_version"] != 1:
        raise RoleBenchError(f"task {task_id} schema_version must be 1")
    normalized["schema_version"] = 1
    for key in ("pair_id", "variant", "workspace"):
        if key in normalized and (not isinstance(normalized[key], str) or not normalized[key]):
            raise RoleBenchError(f"task {task_id} {key} must be a non-empty string")
    if "max_output_tokens" in normalized and (not isinstance(normalized["max_output_tokens"], int) or isinstance(normalized["max_output_tokens"], bool) or normalized["max_output_tokens"] < 1):
        raise RoleBenchError(f"task {task_id} max_output_tokens must be a positive integer")
    if "requires_image" in normalized and not isinstance(normalized["requires_image"], bool):
        raise RoleBenchError(f"task {task_id} requires_image must be a boolean")
    if "image" in normalized:
        normalized["image"] = fixture_string_list(normalized["image"], f"task {task_id} image")
    difficulty = normalized.get("difficulty", {})
    if not isinstance(difficulty, dict):
        raise RoleBenchError(f"task {task_id} difficulty must be an object")
    unknown_axes = sorted(set(difficulty) - DIFFICULTY_AXES - {"level"})
    if unknown_axes:
        raise RoleBenchError(f"task {task_id} difficulty contains unknown axis: {', '.join(unknown_axes)}")
    for key, value in difficulty.items():
        maximum = 5 if key == "level" else 3
        if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value <= maximum:
            raise RoleBenchError(f"task {task_id} difficulty.{key} must be an integer from 0 to {maximum}")
    normalized["difficulty"] = dict(difficulty)
    normalized["expected"] = normalize_expected(normalized.get("expected"), task_id)
    normalized["failure_mode"] = normalize_failure_mode(normalized.get("failure_mode"), task_id)
    invariants = normalized.get("invariants", [])
    if not isinstance(invariants, list):
        raise RoleBenchError(f"task {task_id} invariants must be a list")
    normalized["invariants"] = [normalize_invariant(item, task_id, index) for index, item in enumerate(invariants, 1)]
    check_ids = [item["id"] for item in normalized["invariants"]]
    if len(set(check_ids)) != len(check_ids):
        raise RoleBenchError(f"task {task_id} contains duplicate invariant ids")
    if "dimension_targets" in normalized:
        normalized["dimension_targets"] = fixture_dimensions(normalized["dimension_targets"], f"task {task_id} dimension_targets")
    else:
        derived = [dimension for item in normalized["invariants"] for dimension in item["dimensions"]]
        if normalized["expected"]:
            derived.extend(["correctness", "instruction_following"])
        if normalized["failure_mode"]:
            derived.extend(["failure_handling", "honesty"])
        normalized["dimension_targets"] = list(dict.fromkeys(derived))
    return normalized


def task_command_check_count(tasks: list[dict[str, Any]]) -> int:
    return sum(invariant.get("type") == "command" for task in tasks for invariant in task.get("invariants", []))


def task_command_argvs(tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {"task_id": task["id"], "check_id": invariant["id"], "argv": invariant["argv"]}
        for task in tasks
        for invariant in task.get("invariants", [])
        if invariant.get("type") == "command"
    ]


def load_tasks(args: argparse.Namespace, roles: list[str] | None = None) -> list[dict[str, Any]]:
    args.recent_work = None
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
            if isinstance(payload, dict) and payload.get("schema_version", 1) != 1:
                raise RoleBenchError(f"unsupported task schema_version in {path}: {payload.get('schema_version')!r}")
            tasks = payload if isinstance(payload, list) else payload.get("tasks", []) if isinstance(payload, dict) else []
    else:
        if getattr(args, "task_source", "core") == "recent":
            if not roles:
                raise RoleBenchError("recent task source requires selected roles")
            recent_work = collect_recent_workspaces(args.recent_workspace, args.recent_days, args.recent_commits)
            args.recent_work = recent_work
            tasks = build_recent_tasks(roles, recent_work)
        else:
            tasks = CORE_TASKS if args.suite == "core" else []
    if not tasks:
        raise RoleBenchError("task suite is empty; provide --tasks-file or use --suite core")
    normalized = []
    seen = set()
    for task in tasks:
        if not isinstance(task, dict) or not task.get("id") or not task.get("role") or not task.get("prompt"):
            raise RoleBenchError("each task must contain id, role, and prompt")
        normalized_task = normalize_task(task)
        if normalized_task["id"] in seen:
            raise RoleBenchError(f"duplicate task id: {normalized_task['id']}")
        seen.add(normalized_task["id"])
        normalized.append(normalized_task)
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
            "candidate_variant": None,
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
                "candidate_variant": candidate_identity(candidate),
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
                "candidate_variant": candidate_identity(candidate),
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


def filesystem_entry_type(mode: int) -> str:
    if stat.S_ISFIFO(mode):
        return "fifo"
    if stat.S_ISSOCK(mode):
        return "socket"
    if stat.S_ISCHR(mode):
        return "character_device"
    if stat.S_ISBLK(mode):
        return "block_device"
    return "other"


def snapshot_workspace(workspace: Path) -> dict[str, Any]:
    files: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    incomplete: list[str] = []

    def record_walk_error(exc: OSError) -> None:
        errors.append(f"{exc.filename or workspace}: {exc}")

    for root_value, directories, filenames in os.walk(workspace, followlinks=False, onerror=record_walk_error):
        root = Path(root_value)
        kept_directories = []
        for name in sorted(directories):
            path = root / name
            relative = path.relative_to(workspace).as_posix()
            if path.is_symlink():
                try:
                    files[relative] = {"type": "symlink", "target": os.readlink(path), "mode": path.lstat().st_mode & 0o7777}
                except OSError as exc:
                    errors.append(f"{relative}: {exc}")
                continue
            try:
                files[relative] = {"type": "directory", "mode": path.stat().st_mode & 0o7777}
            except OSError as exc:
                errors.append(f"{relative}: {exc}")
            kept_directories.append(name)
        directories[:] = kept_directories
        for name in sorted(filenames):
            path = root / name
            relative = path.relative_to(workspace).as_posix()
            try:
                path_stat = path.lstat()
                if stat.S_ISLNK(path_stat.st_mode):
                    files[relative] = {"type": "symlink", "target": os.readlink(path), "mode": path_stat.st_mode & 0o7777}
                    continue
                if not stat.S_ISREG(path_stat.st_mode):
                    files[relative] = {"type": filesystem_entry_type(path_stat.st_mode), "mode": path_stat.st_mode & 0o7777}
                    continue
                flags = os.O_RDONLY | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_NOFOLLOW", 0)
                descriptor = os.open(path, flags)
                digest = hashlib.sha256()
                chunks: list[bytes] = []
                with os.fdopen(descriptor, "rb") as handle:
                    opened_stat = os.fstat(handle.fileno())
                    if not stat.S_ISREG(opened_stat.st_mode):
                        files[relative] = {"type": filesystem_entry_type(opened_stat.st_mode), "mode": opened_stat.st_mode & 0o7777}
                        continue
                    size = opened_stat.st_size
                    if size <= SNAPSHOT_HASH_LIMIT:
                        remaining = size
                        while remaining:
                            chunk = handle.read(min(65536, remaining))
                            if not chunk:
                                break
                            digest.update(chunk)
                            remaining -= len(chunk)
                            if size <= SNAPSHOT_TEXT_LIMIT and ".git" not in Path(relative).parts:
                                chunks.append(chunk)
                        final_stat = os.fstat(handle.fileno())
                        stable = (
                            remaining == 0
                            and final_stat.st_size == size
                            and final_stat.st_mtime_ns == opened_stat.st_mtime_ns
                            and final_stat.st_ctime_ns == opened_stat.st_ctime_ns
                        )
                        entry: dict[str, Any] = {"type": "file", "size": size, "sha256": digest.hexdigest(), "hash_complete": stable, "mode": opened_stat.st_mode & 0o7777}
                        if not stable:
                            incomplete.append(relative)
                    else:
                        sample_size = 1024 * 1024
                        digest.update(handle.read(sample_size))
                        handle.seek(max(sample_size, size - sample_size))
                        digest.update(handle.read(sample_size))
                        entry = {"type": "file", "size": size, "sample_sha256": digest.hexdigest(), "hash_complete": False, "mode": opened_stat.st_mode & 0o7777}
                        incomplete.append(relative)
                if size <= SNAPSHOT_TEXT_LIMIT and entry.get("hash_complete") and ".git" not in Path(relative).parts:
                    try:
                        entry["text"] = b"".join(chunks).decode("utf-8")
                    except UnicodeDecodeError:
                        pass
                files[relative] = entry
            except OSError as exc:
                errors.append(f"{relative}: {exc}")
    return {"complete": not errors and not incomplete, "files": files, "errors": errors, "incomplete": incomplete}


def public_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
    return {
        "complete": snapshot["complete"],
        "errors": snapshot["errors"],
        "incomplete": snapshot.get("incomplete", []),
        "files": {
            path: {key: value for key, value in entry.items() if key != "text"}
            for path, entry in snapshot["files"].items()
        },
    }


def diff_workspace_snapshots(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    before_files = before["files"]
    after_files = after["files"]
    created = sorted(set(after_files) - set(before_files))
    deleted = sorted(set(before_files) - set(after_files))
    modified = sorted(
        path for path in set(before_files) & set(after_files)
        if {key: value for key, value in before_files[path].items() if key != "text"}
        != {key: value for key, value in after_files[path].items() if key != "text"}
    )
    patches: list[str] = []
    text_unavailable: list[str] = []
    patch_size = 0
    patch_truncated = False
    for path in created + deleted + modified:
        old_entry = before_files.get(path, {})
        new_entry = after_files.get(path, {})
        old_present = bool(old_entry)
        new_present = bool(new_entry)
        old_text = old_entry.get("text") if old_present else ""
        new_text = new_entry.get("text") if new_present else ""
        if (old_present and not isinstance(old_text, str)) or (new_present and not isinstance(new_text, str)):
            if old_entry.get("type") == "file" or new_entry.get("type") == "file":
                text_unavailable.append(path)
            continue
        old_lines = (old_text or "").splitlines(keepends=True)
        new_lines = (new_text or "").splitlines(keepends=True)
        file_patch = "".join(difflib.unified_diff(old_lines, new_lines, fromfile=f"a/{path}" if path not in created else "/dev/null", tofile=f"b/{path}" if path not in deleted else "/dev/null"))
        remaining = WORKSPACE_PATCH_LIMIT - patch_size
        if len(file_patch) > remaining:
            patches.append(file_patch[:remaining])
            patch_truncated = True
            break
        patches.append(file_patch)
        patch_size += len(file_patch)
    patch = "".join(patches)
    if patch_truncated:
        patch += "\n... workspace patch truncated by RoleBench ...\n"
    return {
        "complete": bool(before["complete"] and after["complete"]),
        "errors": list(before["errors"]) + list(after["errors"]),
        "created": created,
        "deleted": deleted,
        "modified": modified,
        "changed_paths": sorted(created + deleted + modified),
        "patch": patch,
        "patch_truncated": patch_truncated,
        "text_unavailable": text_unavailable,
    }


def evidence_text(value: Any) -> str:
    text = str(value).replace("\n", "\\n")
    return text if len(text) <= EVIDENCE_TEXT_LIMIT else text[:EVIDENCE_TEXT_LIMIT] + "..."


def evaluation_check(check_id: str, kind: str, status: str, evidence: str, dimensions: list[str], hard_gate: bool) -> dict[str, Any]:
    return {
        "id": check_id,
        "type": kind,
        "status": status,
        "evidence": evidence_text(evidence),
        "evaluator": "deterministic",
        "dimensions": dimensions,
        "hard_gate": hard_gate,
    }


def regex_check(pattern: str, text: str, should_match: bool) -> tuple[str, str]:
    matched, error = bounded_regex_search(pattern, text)
    if matched is None:
        return "unknown", error or "regex evidence unavailable"
    passed = matched if should_match else not matched
    expectation = "matched" if should_match else "did not match"
    return ("pass" if passed else "fail"), f"pattern {expectation}: {pattern!r}"


def expected_output_checks(task: dict[str, Any], output: str, execution_completed: bool = True) -> list[dict[str, Any]]:
    expected = task.get("expected", {})
    checks: list[dict[str, Any]] = []
    dimensions = ["correctness", "instruction_following"]
    hard_gate = True
    unavailable = "execution did not complete; output contract not evaluated"
    if "output_exact" in expected:
        wanted = expected["output_exact"]
        status = "unknown" if not execution_completed else ("pass" if output == wanted else "fail")
        detail = unavailable if not execution_completed else f"expected={wanted!r}; actual={output!r}"
        checks.append(evaluation_check("expected-output-exact", "output_exact", status, detail, dimensions, hard_gate))
    for index, wanted in enumerate(expected.get("output_contains", []), 1):
        status = "unknown" if not execution_completed else ("pass" if wanted in output else "fail")
        detail = unavailable if not execution_completed else f"required substring={wanted!r}"
        checks.append(evaluation_check(f"expected-output-contains-{index:03d}", "output_contains", status, detail, dimensions, hard_gate))
    for index, pattern in enumerate(expected.get("output_regex", []), 1):
        status, detail = ("unknown", unavailable) if not execution_completed else regex_check(pattern, output, True)
        checks.append(evaluation_check(f"expected-output-regex-{index:03d}", "output_regex", status, detail, dimensions, hard_gate))
    for index, pattern in enumerate(expected.get("output_not_regex", []), 1):
        status, detail = ("unknown", unavailable) if not execution_completed else regex_check(pattern, output, False)
        checks.append(evaluation_check(f"expected-output-not-regex-{index:03d}", "output_not_regex", status, detail, ["honesty", "instruction_following"], True))
    if "output_json_equals" in expected:
        if not execution_completed:
            status, detail = "unknown", unavailable
        elif len(output) > JSON_CHECK_TEXT_LIMIT:
            status, detail = "unknown", f"JSON output exceeds {JSON_CHECK_TEXT_LIMIT} characters"
        else:
            try:
                actual = json.loads(output)
                status = "pass" if actual == expected["output_json_equals"] else "fail"
                detail = f"expected={expected['output_json_equals']!r}; actual={actual!r}"
            except json.JSONDecodeError as exc:
                status, detail = "fail", f"output is not JSON: {exc}"
            except RecursionError as exc:
                status, detail = "unknown", f"JSON nesting exceeds evaluator limit: {exc}"
        checks.append(evaluation_check("expected-output-json", "output_json_equals", status, detail, dimensions, hard_gate))
    if "max_output_chars" in expected:
        maximum = expected["max_output_chars"]
        status = "unknown" if not execution_completed else ("pass" if len(output) <= maximum else "fail")
        detail = unavailable if not execution_completed else f"actual={len(output)}; maximum={maximum}"
        checks.append(evaluation_check("expected-output-length", "max_output_chars", status, detail, ["instruction_following"], False))
    return checks


def safe_workspace_path(workspace: Path, relative: str) -> Path:
    candidate = workspace / relative
    try:
        candidate.resolve(strict=False).relative_to(workspace.resolve())
    except ValueError as exc:
        raise RoleBenchError(f"fixture path escapes task workspace: {relative!r}") from exc
    return candidate


def file_invariant_check(invariant: dict[str, Any], workspace: Path) -> tuple[str, str]:
    kind = invariant["type"]
    path = safe_workspace_path(workspace, invariant["path"])
    if kind == "file_exists":
        return ("pass", f"exists: {invariant['path']}") if path.is_file() else ("fail", f"missing file: {invariant['path']}")
    if kind == "file_not_exists":
        return ("pass", f"absent: {invariant['path']}") if not os.path.lexists(path) else ("fail", f"unexpected path exists: {invariant['path']}")
    try:
        path_stat = path.lstat()
    except OSError:
        return "fail", f"missing file: {invariant['path']}"
    if not stat.S_ISREG(path_stat.st_mode):
        return "fail", f"not a regular file: {invariant['path']}"
    if path_stat.st_size > FILE_CHECK_TEXT_LIMIT:
        return "unknown", f"file exceeds evaluator limit of {FILE_CHECK_TEXT_LIMIT} bytes: {invariant['path']}"
    try:
        flags = os.O_RDONLY | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        with os.fdopen(descriptor, "rb") as handle:
            if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
                return "fail", f"not a regular file: {invariant['path']}"
            content = handle.read(FILE_CHECK_TEXT_LIMIT + 1)
        if len(content) > FILE_CHECK_TEXT_LIMIT:
            return "unknown", f"file grew beyond evaluator limit of {FILE_CHECK_TEXT_LIMIT} bytes: {invariant['path']}"
        text = content.decode("utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        return "fail", f"cannot read UTF-8 file {invariant['path']}: {exc}"
    if kind == "file_content_exact":
        return ("pass", f"exact content matched: {invariant['path']}") if text == invariant["value"] else ("fail", f"expected={invariant['value']!r}; actual={text!r}")
    if kind == "file_contains":
        missing = [value for value in invariant["values"] if value not in text]
        return ("pass", f"required content found: {invariant['path']}") if not missing else ("fail", f"missing substrings: {missing!r}")
    if kind == "file_matches":
        return regex_check(invariant["pattern"], text, True)
    try:
        actual = json.loads(text)
    except json.JSONDecodeError as exc:
        return "fail", f"file is not JSON: {exc}"
    except RecursionError as exc:
        return "unknown", f"JSON nesting exceeds evaluator limit: {exc}"
    return ("pass", f"JSON matched: {invariant['path']}") if actual == invariant["value"] else ("fail", f"expected={invariant['value']!r}; actual={actual!r}")


def run_command_invariant(invariant: dict[str, Any], workspace: Path, allowed: bool) -> tuple[str, str]:
    if not allowed:
        return "unknown", "command check not enabled; rerun with --allow-check-command after reviewing argv"
    try:
        with tempfile.TemporaryFile() as stdout_file, tempfile.TemporaryFile() as stderr_file:
            process = subprocess.Popen(
                invariant["argv"],
                cwd=workspace,
                stdin=subprocess.DEVNULL,
                stdout=stdout_file,
                stderr=stderr_file,
                env=minimal_command_environment(workspace),
            )
            try:
                returncode = process.wait(timeout=invariant["timeout"])
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
                return "fail", f"command timed out after {invariant['timeout']}s"
            stdout_file.seek(0)
            stderr_file.seek(0)
            stdout_bytes = stdout_file.read(COMMAND_OUTPUT_LIMIT + 1)
            stderr_bytes = stderr_file.read(COMMAND_OUTPUT_LIMIT + 1)
    except (OSError, ValueError) as exc:
        return "fail", f"cannot run command: {exc}"
    stdout_truncated = len(stdout_bytes) > COMMAND_OUTPUT_LIMIT
    stderr_truncated = len(stderr_bytes) > COMMAND_OUTPUT_LIMIT
    stdout = stdout_bytes[:COMMAND_OUTPUT_LIMIT].decode("utf-8", errors="replace")
    stderr = stderr_bytes[:COMMAND_OUTPUT_LIMIT].decode("utf-8", errors="replace")
    failures = []
    unknowns = []
    if returncode != invariant["expected_exit_code"]:
        failures.append(f"exit={returncode}, expected={invariant['expected_exit_code']}")
    for value in invariant.get("stdout_contains", []):
        if value not in stdout:
            (unknowns if stdout_truncated else failures).append(f"stdout missing {value!r}")
    for value in invariant.get("stderr_contains", []):
        if value not in stderr:
            (unknowns if stderr_truncated else failures).append(f"stderr missing {value!r}")
    if failures:
        return "fail", "; ".join(failures)
    if unknowns:
        return "unknown", "; ".join(unknowns) + "; command output was truncated"
    suffix = "; output truncated" if stdout_truncated or stderr_truncated else ""
    return "pass", f"exit={returncode}; stdout={stdout!r}; stderr={stderr!r}{suffix}"


def path_glob_matches(path: str, pattern: str) -> bool:
    path_parts = path.split("/") if path else []
    pattern_parts = pattern.split("/") if pattern else []

    def match(path_index: int, pattern_index: int) -> bool:
        if pattern_index == len(pattern_parts):
            return path_index == len(path_parts)
        if pattern_parts[pattern_index] == "**":
            return match(path_index, pattern_index + 1) or (
                path_index < len(path_parts) and match(path_index + 1, pattern_index)
            )
        return (
            path_index < len(path_parts)
            and fnmatch.fnmatchcase(path_parts[path_index], pattern_parts[pattern_index])
            and match(path_index + 1, pattern_index + 1)
        )

    return match(0, 0)


def baseline_symlink_for_path(before_snapshot: dict[str, Any] | None, relative: str) -> bool:
    if not before_snapshot:
        return False
    files = before_snapshot.get("files", {})
    parts = relative.split("/")
    return any(files.get("/".join(parts[:index]), {}).get("type") == "symlink" for index in range(1, len(parts) + 1))


def deterministic_checks(task: dict[str, Any], result: dict[str, Any], workspace: Path | None, workspace_diff: dict[str, Any] | None, allow_check_command: bool, execution_completed: bool = True, before_snapshot: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    output = result.get("output", "")
    checks = expected_output_checks(task, output, execution_completed)
    failure_mode = task.get("failure_mode", {})
    unavailable = "execution did not complete; output contract not evaluated"
    for index, pattern in enumerate(failure_mode.get("required_output_regex", []), 1):
        status, detail = ("unknown", unavailable) if not execution_completed else regex_check(pattern, output, True)
        checks.append(evaluation_check(f"failure-required-{index:03d}", "required_output_regex", status, detail, ["failure_handling"], False))
    for index, pattern in enumerate(failure_mode.get("forbidden_output_regex", []), 1):
        status, detail = ("unknown", unavailable) if not execution_completed else regex_check(pattern, output, False)
        checks.append(evaluation_check(f"failure-forbidden-{index:03d}", "forbidden_output_regex", status, detail, ["honesty"], True))
    for invariant in task.get("invariants", []):
        kind = invariant["type"]
        try:
            if workspace is None:
                status, evidence = "unknown", "workspace evidence unavailable"
            elif kind in {"allowed_changed_paths", "forbidden_changed_paths"}:
                if not workspace_diff or not workspace_diff["complete"]:
                    status, evidence = "unknown", "workspace snapshot incomplete"
                else:
                    changed = workspace_diff["changed_paths"]
                    if kind == "allowed_changed_paths":
                        offending = [path for path in changed if not any(path_glob_matches(path, pattern) for pattern in invariant["patterns"])]
                        status = "pass" if not offending else "fail"
                        evidence = f"changed={changed!r}; outside allowed patterns={offending!r}"
                    else:
                        offending = [path for path in changed if any(path_glob_matches(path, pattern) for pattern in invariant["patterns"])]
                        status = "pass" if not offending else "fail"
                        evidence = f"changed={changed!r}; forbidden matches={offending!r}"
            elif kind == "command":
                status, evidence = run_command_invariant(invariant, workspace, allow_check_command)
            else:
                status, evidence = file_invariant_check(invariant, workspace)
        except RoleBenchError as exc:
            if "path escapes" in str(exc) and baseline_symlink_for_path(before_snapshot, invariant.get("path", "")):
                status, evidence = "unknown", f"baseline fixture symlink prevents safe evaluation: {exc}"
            else:
                status, evidence = "fail", str(exc)
        checks.append(evaluation_check(invariant["id"], kind, status, evidence, invariant["dimensions"], invariant["hard_gate"]))
    return checks


def aggregate_dimensions(task: dict[str, Any], checks: list[dict[str, Any]]) -> dict[str, Any]:
    dimensions: dict[str, Any] = {}
    for dimension in task.get("dimension_targets", []):
        relevant = [check for check in checks if dimension in check["dimensions"]]
        if not relevant:
            status, score = "unknown", None
            evidence = ["no deterministic evidence for this dimension"]
        elif any(check["status"] == "fail" for check in relevant):
            status, score = "fail", 0
            evidence = [check["evidence"] for check in relevant if check["status"] == "fail"]
        elif any(check["status"] == "unknown" for check in relevant):
            status, score = "unknown", None
            evidence = [check["evidence"] for check in relevant if check["status"] == "unknown"]
        else:
            status, score = "pass", 2
            evidence = [check["evidence"] for check in relevant]
        dimensions[dimension] = {
            "status": status,
            "score": score,
            "evidence": evidence[:5],
            "evaluator": "deterministic" if relevant else "none",
            "hard_gate": dimension in HARD_GATE_DIMENSIONS,
        }
    return dimensions


def build_evaluation(task: dict[str, Any], result: dict[str, Any], workspace: Path | None, workspace_diff: dict[str, Any] | None, allow_check_command: bool, artifacts: dict[str, str] | None, before_snapshot: dict[str, Any] | None = None) -> dict[str, Any]:
    if result.get("error_type") == "timeout":
        execution_status = "timeout"
    elif result.get("status") == "pass":
        execution_status = "completed"
    else:
        execution_status = "failed"
    checks = deterministic_checks(task, result, workspace, workspace_diff, allow_check_command, execution_status == "completed", before_snapshot)
    dimensions = aggregate_dimensions(task, checks)
    hard_checks = [check for check in checks if check["hard_gate"]]
    hard_dimensions = [value for name, value in dimensions.items() if name in HARD_GATE_DIMENSIONS]
    if not hard_checks and not hard_dimensions:
        hard_gate_status = "unknown"
    elif any(check["status"] == "fail" for check in hard_checks) or any(value["status"] == "fail" for value in hard_dimensions):
        hard_gate_status = "fail"
    elif execution_status != "completed" or any(check["status"] == "unknown" for check in hard_checks) or any(value["status"] == "unknown" for value in hard_dimensions):
        hard_gate_status = "unknown"
    else:
        hard_gate_status = "pass"
    if execution_status != "completed" or any(check["status"] == "fail" for check in checks):
        outcome = "fail"
    elif not checks or any(check["status"] == "unknown" for check in checks) or any(value["status"] == "unknown" for value in dimensions.values()):
        outcome = "unknown"
    else:
        outcome = "pass"
    return {
        "schema_version": 1,
        "evaluation_status": "completed",
        "timestamp": now_iso(),
        "assignment_id": result["assignment_id"],
        "task_id": result["task_id"],
        "role": result["role"],
        "model": result["model"],
        "thinking": result.get("thinking"),
        "replicate": result["replicate"],
        "execution_status": execution_status,
        "outcome": outcome,
        "hard_gate_status": hard_gate_status,
        "checks": checks,
        "dimensions": dimensions,
        "artifacts": artifacts or {},
    }


def failed_evaluation(task: dict[str, Any], result: dict[str, Any], error: Exception | str, artifacts: dict[str, str] | None = None) -> dict[str, Any]:
    if result.get("error_type") == "timeout":
        execution_status = "timeout"
    elif result.get("status") == "pass":
        execution_status = "completed"
    else:
        execution_status = "failed"
    detail = evidence_text(error)
    dimensions = {
        dimension: {
            "status": "unknown",
            "score": None,
            "evidence": [f"deterministic evaluator failed: {detail}"],
            "evaluator": "deterministic",
            "hard_gate": dimension in HARD_GATE_DIMENSIONS,
        }
        for dimension in task.get("dimension_targets", [])
    }
    return {
        "schema_version": 1,
        "evaluation_status": "failed",
        "timestamp": now_iso(),
        "assignment_id": result.get("assignment_id"),
        "task_id": result.get("task_id", task.get("id")),
        "role": result.get("role", task.get("role")),
        "model": result.get("model", ""),
        "thinking": result.get("thinking"),
        "replicate": result.get("replicate"),
        "execution_status": execution_status,
        "outcome": "fail" if execution_status != "completed" else "unknown",
        "hard_gate_status": "unknown",
        "checks": [],
        "dimensions": dimensions,
        "artifacts": artifacts or {},
        "errors": [detail],
    }


def write_workspace_artifacts(output_dir: Path, artifacts_root: Path, identity: str, before: dict[str, Any], after: dict[str, Any], workspace_diff: dict[str, Any]) -> dict[str, str]:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", identity).strip("-")[:80] or "cell"
    suffix = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:10]
    artifact_dir = artifacts_root / f"{slug}-{suffix}"
    artifact_dir.mkdir(parents=True, exist_ok=False)
    before_path = artifact_dir / "workspace-before.json"
    after_path = artifact_dir / "workspace-after.json"
    diff_path = artifact_dir / "workspace-diff.json"
    patch_path = artifact_dir / "workspace.patch"
    write_json(before_path, public_snapshot(before))
    write_json(after_path, public_snapshot(after))
    write_json(diff_path, {key: value for key, value in workspace_diff.items() if key != "patch"})
    patch_path.write_text(workspace_diff["patch"], encoding="utf-8")
    return {
        "workspace_before": str(before_path.relative_to(output_dir)),
        "workspace_after": str(after_path.relative_to(output_dir)),
        "workspace_diff": str(diff_path.relative_to(output_dir)),
        "workspace_patch": str(patch_path.relative_to(output_dir)),
    }


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
                "thinking": assignment.get("thinking", {}).get(task["role"]),
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


def fixed_judge_dimensions(task: dict[str, Any]) -> list[str]:
    return list(dict.fromkeys(task.get("dimension_targets", [])))


def build_fixed_judge_prompt(task: dict[str, Any], result: dict[str, Any], deterministic_evaluation: dict[str, Any]) -> str:
    model_output = decode_subprocess_text(result.get("output", ""))
    error_output = evidence_text(result.get("error", ""))
    task_view = {
        key: value
        for key, value in task.items()
        if key not in {"workspace", "image"}
    }
    evidence_view = {
        "execution_status": deterministic_evaluation.get("execution_status"),
        "outcome": deterministic_evaluation.get("outcome"),
        "hard_gate_status": deterministic_evaluation.get("hard_gate_status"),
        "checks": [
            {
                "type": check.get("type"),
                "status": check.get("status"),
                "evidence": check.get("evidence"),
                "dimensions": check.get("dimensions"),
            }
            for check in deterministic_evaluation.get("checks", [])
        ],
    }
    payload = {
        "task": task_view,
        "role": task.get("role"),
        "execution": {
            "status": result.get("execution_status") or deterministic_evaluation.get("execution_status"),
            "output": model_output if len(model_output) <= JUDGE_INPUT_LIMIT else model_output[:JUDGE_INPUT_LIMIT] + "...",
            "error": error_output,
        },
        "deterministic_evidence": evidence_view,
    }
    return (
        "You are the fixed semantic evaluator for OMP RoleBench. "
        f"Rubric version: {FIXED_JUDGE_RUBRIC_VERSION}. Return JSON only, with no markdown. "
        "Evaluate the work against the task and available evidence, never infer hidden actions, "
        "and mark unsupported claims unknown. For each requested dimension return status pass, fail, "
        "or unknown; score must be 0 for fail, 2 for pass, and null for unknown; evidence must be a "
        "short list of observable reasons. Do not identify or guess the provider model.\n\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
    )


def estimate_fixed_judge(roles: list[str], assignments: list[dict[str, Any]], tasks: list[dict[str, Any]], runs: int, judge_model: str | None, pricing: dict[str, dict[str, float]]) -> dict[str, Any]:
    if not judge_model:
        return {"enabled": False, "model": None, "calls": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0, "cost_status": "not-requested", "by_assignment": []}
    rows = []
    calls = 0
    input_tokens = 0
    output_tokens = 0
    known_cost = 0.0
    cost_known = True
    selector = parse_selector(judge_model)["selector"]
    for assignment in assignments:
        applicable = [task for task in tasks if task_for_assignment(task, assignment)]
        for task in applicable:
            max_evidence = evidence_text("x" * (EVIDENCE_TEXT_LIMIT + 1))
            skeleton_checks = deterministic_checks(
                task,
                {"output": "", "execution_status": "completed"},
                None,
                None,
                False,
                True,
            )
            skeleton_checks = [
                {**check, "evidence": max_evidence}
                for check in skeleton_checks
            ]
            skeleton_evaluation = {
                "execution_status": "completed",
                "outcome": "unknown",
                "hard_gate_status": "unknown",
                "checks": skeleton_checks,
            }
            estimate_prompt = build_fixed_judge_prompt(
                task,
                {"execution_status": "completed", "output": "x" * (JUDGE_INPUT_LIMIT + 1), "error": "x" * (EVIDENCE_TEXT_LIMIT + 1)},
                skeleton_evaluation,
            )
            task_input = max(1, math.ceil(len(estimate_prompt) / 4))
            task_calls = runs
            calls += task_calls
            input_tokens += task_input * task_calls
            output_tokens += FIXED_JUDGE_OUTPUT_TOKENS * task_calls
            rates = pricing.get(selector, {})
            if "input_per_million" in rates and "output_per_million" in rates:
                known_cost += task_calls * (task_input * rates["input_per_million"] + FIXED_JUDGE_OUTPUT_TOKENS * rates["output_per_million"]) / 1_000_000
            else:
                cost_known = False
            rows.append({
                "assignment_id": assignment["id"],
                "role": task["role"],
                "calls": task_calls,
                "input_tokens": task_input * task_calls,
                "output_tokens": FIXED_JUDGE_OUTPUT_TOKENS * task_calls,
            })
    return {
        "enabled": True,
        "model": judge_model,
        "calls": calls,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": round(known_cost, 8) if cost_known else None,
        "cost_status": "estimated" if cost_known else "unknown",
        "by_assignment": rows,
    }


def unknown_judge_dimensions(dimensions: list[str], evidence: str) -> dict[str, Any]:
    return {
        dimension: {
            "status": "unknown",
            "score": None,
            "evidence": [evidence],
            "evaluator": "fixed_judge",
            "hard_gate": dimension in HARD_GATE_DIMENSIONS,
        }
        for dimension in dimensions
    }


def build_judge_evaluation(task: dict[str, Any], result: dict[str, Any], judge_result: dict[str, Any], dimensions_payload: dict[str, Any] | None, error: str | None = None) -> dict[str, Any]:
    dimensions = fixed_judge_dimensions(task)
    if error or not isinstance(dimensions_payload, dict):
        dimensions_value = unknown_judge_dimensions(dimensions, error or "fixed judge returned no dimension object")
    else:
        dimensions_value = {}
        for dimension in dimensions:
            item = dimensions_payload.get(dimension)
            if not isinstance(item, dict):
                dimensions_value[dimension] = unknown_judge_dimensions([dimension], "fixed judge omitted this dimension")[dimension]
                continue
            status = item.get("status")
            score = item.get("score")
            if status not in {"pass", "fail", "unknown"}:
                status = "unknown"
            if status == "pass":
                score = 2
            elif status == "fail":
                score = 0
            else:
                score = None
            evidence = item.get("evidence", [])
            if isinstance(evidence, str):
                evidence = [evidence]
            if not isinstance(evidence, list) or not all(isinstance(value, str) for value in evidence):
                evidence = ["fixed judge evidence was malformed"]
                status, score = "unknown", None
            dimensions_value[dimension] = {
                "status": status,
                "score": score,
                "evidence": [evidence_text(value) for value in evidence[:5]],
                "evaluator": "fixed_judge",
                "hard_gate": dimension in HARD_GATE_DIMENSIONS,
            }
    values = list(dimensions_value.values())
    hard_values = [value for value in values if value["hard_gate"]]
    if any(value["status"] == "fail" for value in hard_values):
        hard_gate_status = "fail"
    elif not hard_values or any(value["status"] == "unknown" for value in hard_values):
        hard_gate_status = "unknown"
    else:
        hard_gate_status = "pass"
    if any(value["status"] == "fail" for value in values):
        outcome = "fail"
    elif not values or any(value["status"] == "unknown" for value in values):
        outcome = "unknown"
    else:
        outcome = "pass"
    return {
        "schema_version": FIXED_JUDGE_SCHEMA_VERSION,
        "evaluation_status": "failed" if error else "completed",
        "evaluator": "fixed_judge",
        "rubric_version": FIXED_JUDGE_RUBRIC_VERSION,
        "timestamp": now_iso(),
        "assignment_id": result.get("assignment_id"),
        "task_id": result.get("task_id", task.get("id")),
        "role": result.get("role", task.get("role")),
        "model": result.get("model", ""),
        "thinking": result.get("thinking"),
        "replicate": result.get("replicate"),
        "execution_status": result.get("execution_status"),
        "outcome": outcome,
        "hard_gate_status": hard_gate_status,
        "dimensions": dimensions_value,
        "summary": dimensions_payload.get("summary") if isinstance(dimensions_payload, dict) and isinstance(dimensions_payload.get("summary"), str) else None,
        "judge_result_id": judge_result.get("judge_result_id"),
        "errors": [error] if error else [],
    }


def run_fixed_judge(omp_bin: str, judge_model: str, task: dict[str, Any], result: dict[str, Any], deterministic_evaluation: dict[str, Any], timeout: int = FIXED_JUDGE_TIMEOUT) -> tuple[dict[str, Any], dict[str, Any]]:
    judge_result_id = f"{result.get('assignment_id', '')}:{result.get('task_id', '')}:{result.get('replicate', '')}"
    raw: dict[str, Any] = {
        "schema_version": FIXED_JUDGE_SCHEMA_VERSION,
        "judge_result_id": judge_result_id,
        "timestamp": now_iso(),
        "assignment_id": result.get("assignment_id"),
        "task_id": result.get("task_id", task.get("id")),
        "role": result.get("role", task.get("role")),
        "model": judge_model,
        "replicate": result.get("replicate"),
        "evaluator": "fixed_judge",
        "rubric_version": FIXED_JUDGE_RUBRIC_VERSION,
        "provider_call_started": False,
        "status": "error",
        "execution_status": "failed",
    }
    try:
        prompt = build_fixed_judge_prompt(task, result, deterministic_evaluation)
        judge_selector = parse_selector(judge_model)
        raw["model"] = judge_selector["selector"]
        raw["thinking"] = judge_selector.get("thinking")
        with tempfile.TemporaryDirectory(prefix="omp-rolebench-judge-") as directory:
            judge_command = [
                omp_bin,
                "-p",
                "--mode",
                "json",
                "--no-session",
                "--no-title",
                "--model",
                judge_selector["selector"],
                "--cwd",
            ]
            if judge_selector.get("thinking"):
                judge_command.extend(["--thinking", judge_selector["thinking"]])
            judge_command.extend([directory, "--auto-approve", prompt])
            raw["provider_call_started"] = True
            try:
                completed = subprocess.run(
                    judge_command,
                    check=False,
                    capture_output=True,
                    text=True,
                    errors="replace",
                    timeout=timeout,
                )
            except (OSError, ValueError):
                raw["provider_call_started"] = False
                raise
        raw["exit_code"] = completed.returncode
        raw["stdout"] = decode_subprocess_text(completed.stdout)
        raw["stderr"] = decode_subprocess_text(completed.stderr)
        if completed.returncode != 0:
            raw["error_type"] = "judge_exit"
            raw["error"] = raw["stderr"].strip() or f"fixed judge exited with {completed.returncode}"
            return raw, build_judge_evaluation(task, result, raw, None, raw["error"])
        payload = parse_json_output(raw["stdout"])
        judge_text = extract_output(payload)
        candidate = payload if isinstance(payload, dict) and isinstance(payload.get("dimensions"), dict) else json.loads(judge_text)
        if not isinstance(candidate, dict) or not isinstance(candidate.get("dimensions"), dict):
            raise RoleBenchError("fixed judge output must contain a dimensions object")
        raw["output"] = judge_text
        usage = extract_usage(payload)
        if usage is not None:
            raw["usage"] = usage
        raw["status"] = "pass"
        raw["execution_status"] = "completed"
        return raw, build_judge_evaluation(task, result, raw, candidate["dimensions"])
    except subprocess.TimeoutExpired as exc:
        raw["error_type"] = "judge_timeout"
        raw["error"] = f"fixed judge exceeded {timeout}s"
        raw["stdout"] = decode_subprocess_text(exc.stdout)
        raw["stderr"] = decode_subprocess_text(exc.stderr)
        return raw, build_judge_evaluation(task, result, raw, None, raw["error"])
    except (OSError, RoleBenchError, json.JSONDecodeError, RecursionError, TypeError, ValueError) as exc:
        raw["error_type"] = "judge_invalid"
        raw["error"] = str(exc)
        return raw, build_judge_evaluation(task, result, raw, None, raw["error"])


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
        f"- task source: {manifest.get('task_source', 'core')}",
        f"- task fixtures: {len(manifest['tasks'])}",
        f"- assignments: {estimate['assignments']}",
        f"- requests: {estimate['calls']} ({manifest['runs']} run(s) per task)",
        f"- estimated tokens: input {estimate['input_tokens']}, output {estimate['output_tokens']}",
        f"- estimated model cost: {cost} ({estimate['cost_status']})",
        f"- estimated total cost: {('$%.6f' % estimate['total_cost_usd']) if estimate.get('total_cost_usd') is not None else 'unknown'} ({estimate.get('total_cost_status', estimate['cost_status'])})",
        f"- cost cap: {cap_text}",
        f"- deterministic evaluator: {', '.join(manifest.get('evaluation', {}).get('dimension_targets', [])) or 'none'} (no model cost)",
        f"- trusted command checks: {manifest.get('evaluation', {}).get('command_checks', 0)} ({'enabled' if manifest.get('evaluation', {}).get('allow_check_command') else 'disabled'})",
    ]
    fixed_judge = manifest.get("evaluation", {}).get("fixed_judge", {})
    if fixed_judge.get("enabled"):
        judge_estimate = estimate.get("judge", {})
        judge_tokens = f"tokens input {judge_estimate.get('input_tokens', 0)}, output {judge_estimate.get('output_tokens', 0)}"
        lines.insert(9, f"- fixed judge: {fixed_judge.get('model')} / rubric {fixed_judge.get('rubric_version')} ({judge_estimate.get('calls', 0)} request(s), {judge_tokens}, ${judge_estimate.get('cost_usd'):.6f} estimated)" if judge_estimate.get("cost_usd") is not None else f"- fixed judge: {fixed_judge.get('model')} / rubric {fixed_judge.get('rubric_version')} ({judge_estimate.get('calls', 0)} request(s), {judge_tokens}, cost unknown)")
    if manifest.get("recommend"):
        lines.append(f"- recommendations: enabled after {manifest.get('min_recommendation_runs', 3)} run(s) per task and deterministic/fixed-judge hard gates")
    for item in manifest.get("evaluation", {}).get("command_argvs", []):
        lines.append(f"- command argv [{item['task_id']}/{item['check_id']}]: {json.dumps(item['argv'], ensure_ascii=False)}")
    recent_work = manifest.get("recent_work")
    if manifest.get("task_source") == "recent" and isinstance(recent_work, dict):
        workspaces = recent_work.get("workspaces") or []
        workspace_labels = [
            f"{item['workspace']} [{item.get('source', 'unknown')}]"
            for item in workspaces
            if isinstance(item, dict) and item.get("workspace")
        ]
        if workspace_labels:
            lines.insert(5, f"- recent workspaces ({len(workspace_labels)}): {', '.join(workspace_labels)}")
        warnings = [
            f"{item.get('workspace', 'unknown')}: {item['git_error']}"
            for item in workspaces
            if isinstance(item, dict) and item.get("git_error")
        ]
        if warnings:
            lines.insert(6, "- recent work warnings: " + " | ".join(warnings))
    return "\n".join(lines)


def percentile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(len(ordered) * probability) - 1))
    return ordered[index]


def build_recommendations(
    roles: list[str],
    tasks: list[dict[str, Any]],
    results: list[dict[str, Any]],
    evaluations: list[dict[str, Any]],
    judge_results: list[dict[str, Any]],
    judge_evaluations: list[dict[str, Any]],
    min_runs: int,
) -> dict[str, Any]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for evaluation in evaluations:
        groups[(evaluation.get("role", ""), evaluation.get("model", ""), evaluation.get("thinking") or "")].append(evaluation)
    raw_groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        raw_groups[(result.get("role", ""), result.get("model", ""), result.get("thinking") or "")].append(result)
    judge_by_cell = {
        (evaluation.get("assignment_id"), evaluation.get("task_id"), evaluation.get("replicate")): evaluation
        for evaluation in judge_evaluations
    }
    judge_raw_by_cell = {
        (result.get("assignment_id"), result.get("task_id"), result.get("replicate")): result
        for result in judge_results
    }
    candidates_by_role: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for (role, model, thinking), values in sorted(groups.items()):
        counts = defaultdict(int)
        for value in values:
            counts[value.get("task_id")] += 1
        blockers: list[str] = []
        expected_task_ids = sorted({task["id"] for task in tasks if task.get("role") == role})
        if not expected_task_ids:
            blockers.append("no task fixtures for this role")
        else:
            incomplete_tasks = [task_id for task_id in expected_task_ids if counts.get(task_id, 0) < min_runs]
            if incomplete_tasks:
                blockers.append(f"need at least {min_runs} runs for every task; incomplete={incomplete_tasks!r}")
        deterministic_hard = [value.get("hard_gate_status") for value in values]
        if any(status == "fail" for status in deterministic_hard):
            blockers.append("deterministic hard gate failed")
        elif not deterministic_hard or any(status != "pass" for status in deterministic_hard):
            blockers.append("deterministic hard gate is not fully verified")
        judge_values = [judge_by_cell.get((value.get("assignment_id"), value.get("task_id"), value.get("replicate"))) for value in values]
        judge_values = [value for value in judge_values if value is not None]
        if len(judge_values) < len(values):
            blockers.append("fixed judge evidence is incomplete")
        judge_hard_statuses = [
            dimension.get("status")
            for value in judge_values
            for name, dimension in value.get("dimensions", {}).items()
            if name in HARD_GATE_DIMENSIONS and isinstance(dimension, dict)
        ]
        judge_hard_dimensions = sorted({
            name
            for value in judge_values
            for name, dimension in value.get("dimensions", {}).items()
            if name in HARD_GATE_DIMENSIONS and isinstance(dimension, dict)
        })
        if any(status == "fail" for status in judge_hard_statuses):
            blockers.append("fixed judge hard gate failed")
        elif not judge_hard_statuses or any(status != "pass" for status in judge_hard_statuses):
            blockers.append("fixed judge hard gate is not fully verified")
        scores = [
            dimension.get("score")
            for value in judge_values
            for name, dimension in value.get("dimensions", {}).items()
            if name not in HARD_GATE_DIMENSIONS and isinstance(dimension, dict) and isinstance(dimension.get("score"), (int, float))
        ]
        quality_score = round(sum(scores) / len(scores), 4) if scores else None
        raw_values = raw_groups[(role, model, thinking)]
        latency = [float(value["latency_ms"]) for value in raw_values if isinstance(value.get("latency_ms"), (int, float))]
        model_costs = [cost for value in raw_values if (cost := extract_cost(value.get("usage"))) is not None]
        judge_costs = [
            cost
            for value in judge_values
            if (cost := extract_cost(judge_raw_by_cell.get((value.get("assignment_id"), value.get("task_id"), value.get("replicate")), {}).get("usage"))) is not None
        ]
        model_cost_known = bool(raw_values) and len(model_costs) == len(raw_values)
        judge_cost_known = bool(judge_values) and len(judge_costs) == len(judge_values)
        model_cost = round(sum(model_costs), 8) if model_cost_known else None
        judge_cost = round(sum(judge_costs), 8) if judge_cost_known else None
        total_cost = round(model_cost + judge_cost, 8) if model_cost is not None and judge_cost is not None else None
        evidence_paths = {"results.jsonl", "evaluations.jsonl", "judge_results.jsonl", "judge_evaluations.jsonl"}
        for value in values:
            artifacts = value.get("artifacts", {})
            if isinstance(artifacts, dict):
                evidence_paths.update(path for path in artifacts.values() if isinstance(path, str))
        candidate = {
            "role": role,
            "model": model,
            "thinking": thinking or None,
            "samples": len(values),
            "samples_per_task": dict(counts),
            "deterministic_hard_gate": "pass" if deterministic_hard and all(status == "pass" for status in deterministic_hard) else "blocked",
            "judge_hard_gate": "pass" if judge_hard_statuses and all(status == "pass" for status in judge_hard_statuses) else "blocked",
            "judge_hard_dimensions": judge_hard_dimensions,
            "quality_score": quality_score,
            "p50_latency_ms": percentile(latency, 0.5),
            "model_cost_usd": model_cost,
            "judge_cost_usd": judge_cost,
            "total_cost_usd": total_cost,
            "cost_status": "observed" if total_cost is not None else "unknown",
            "evidence": sorted(evidence_paths),
            "eligible": not blockers,
            "blocked_reasons": blockers,
        }
        candidates_by_role[role].append(candidate)
    role_recommendations = []
    for role in roles:
        candidates = candidates_by_role.get(role, [])
        eligible = [candidate for candidate in candidates if candidate["eligible"]]
        eligible.sort(key=lambda candidate: (
            -(candidate["quality_score"] if candidate["quality_score"] is not None else -1),
            candidate["p50_latency_ms"] if candidate["p50_latency_ms"] is not None else float("inf"),
            candidate["total_cost_usd"] is None,
            candidate["total_cost_usd"] if candidate["total_cost_usd"] is not None else float("inf"),
        ))
        winner = eligible[0] if eligible else None
        role_recommendations.append({
            "role": role,
            "model": winner["model"] if winner else None,
            "thinking": winner["thinking"] if winner else None,
            "status": "recommended" if winner else "blocked",
            "reason": (
                "hard gates passed; highest fixed-judge quality score, then latency and cost"
                if winner
                else "no candidate met the minimum runs and hard-gate requirements"
            ),
            "candidate": winner,
        })
    return {
        "schema_version": RECOMMENDATION_SCHEMA_VERSION,
        "generated_at": now_iso(),
        "evaluator": "deterministic+fixed_judge",
        "min_runs_per_task": min_runs,
        "status": "complete" if all(item["status"] == "recommended" for item in role_recommendations) else "blocked",
        "roles": role_recommendations,
        "candidates": [candidate for role in roles for candidate in candidates_by_role.get(role, [])],
    }


def output_copy_ignore(source: Path, output_dir: Path):
    try:
        relative = output_dir.resolve().relative_to(source.resolve())
    except ValueError:
        return None
    if not relative.parts:
        raise RoleBenchError("benchmark output directory cannot be the task workspace")
    parent = Path(*relative.parts[:-1]) if len(relative.parts) > 1 else Path(".")
    target = relative.parts[-1]

    def ignore(current: str, names: list[str]) -> set[str]:
        try:
            current_relative = Path(current).resolve().relative_to(source.resolve())
        except ValueError:
            return set()
        return {target} if current_relative == parent and target in names else set()

    return ignore


def run_one(omp_bin: str, assignment: dict[str, Any], task: dict[str, Any], replicate: int, output_dir: Path, timeout: int, config_file: Path, executor: str, controller_model: str | None, allow_check_command: bool = False, artifacts_root: Path | None = None) -> dict[str, Any]:
    started = time.perf_counter()
    result: dict[str, Any] = {
        "timestamp": now_iso(),
        "assignment_id": assignment["id"],
        "mode": assignment["mode"],
        "role": task["role"],
        "task_id": task["id"],
        "replicate": replicate,
        "model": assignment["mapping"].get(task["role"], ""),
        "thinking": assignment.get("thinking", {}).get(task["role"]),
        "executor": executor,
        "provider_call_started": False,
        "status": "error",
        "execution_status": "failed",
    }
    if task.get("requires_image") and not task.get("image"):
        result.update({"error_type": "missing_image_fixture", "error": "task requires an image fixture"})
        try:
            result["_evaluation"] = build_evaluation(task, result, None, None, allow_check_command, None)
        except Exception as exc:
            result.setdefault("warnings", []).append({"type": "evaluation", "error": str(exc)})
            result["_evaluation"] = failed_evaluation(task, result, exc)
        result["latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
        return result
    temp_root = Path(tempfile.mkdtemp(prefix="omp-rolebench-"))
    workspace: Path | None = None
    before_snapshot: dict[str, Any] | None = None
    try:
        workspace = temp_root / "workspace"
        source_workspace = task.get("workspace")
        if source_workspace:
            source = Path(source_workspace).expanduser().resolve()
            if not source.is_dir():
                raise RoleBenchError(f"task workspace does not exist: {source}")
            shutil.copytree(source, workspace, ignore=output_copy_ignore(source, output_dir))
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
        before_snapshot = snapshot_workspace(workspace)
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
        result["provider_call_started"] = True
        try:
            completed = subprocess.run(command, check=False, capture_output=True, text=True, errors="replace", timeout=timeout)
        except (OSError, ValueError):
            result["provider_call_started"] = False
            raise
        result["exit_code"] = completed.returncode
        stdout = decode_subprocess_text(completed.stdout)
        stderr = decode_subprocess_text(completed.stderr)
        result["stdout"] = stdout
        result["stderr"] = stderr
        result["command"] = command[:-(1 + len(image_args))]
        if completed.returncode != 0:
            result["error_type"] = "omp_exit"
            result["error"] = stderr.strip() or f"OMP exited with {completed.returncode}"
        else:
            try:
                payload = parse_json_output(stdout)
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
                if result["status"] == "pass":
                    result["execution_status"] = "completed"
            except RoleBenchError as exc:
                result["error_type"] = "malformed_omp_json"
                result["error"] = str(exc)
    except subprocess.TimeoutExpired as exc:
        result["error_type"] = "timeout"
        result["execution_status"] = "timeout"
        result["error"] = f"OMP exceeded {timeout}s"
        result["stdout"] = decode_subprocess_text(exc.stdout)
        result["stderr"] = decode_subprocess_text(exc.stderr)
    except (OSError, RoleBenchError) as exc:
        result["error_type"] = "executor_setup"
        result["error"] = str(exc)
    except Exception as exc:
        result["error_type"] = "executor_internal"
        result["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        result["latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
        workspace_diff = None
        artifacts = None
        evaluation_workspace = workspace if before_snapshot is not None and workspace and workspace.is_dir() else None
        try:
            if evaluation_workspace is not None:
                after_snapshot = snapshot_workspace(evaluation_workspace)
                workspace_diff = diff_workspace_snapshots(before_snapshot, after_snapshot)
                if artifacts_root is not None:
                    identity = f"{assignment['id']}-{task['id']}-run-{replicate}"
                    artifacts = write_workspace_artifacts(output_dir, artifacts_root, identity, before_snapshot, after_snapshot, workspace_diff)
        except Exception as exc:
            warning_type = "artifact_write" if workspace_diff is not None else "workspace_evidence"
            result.setdefault("warnings", []).append({"type": warning_type, "error": str(exc)})
        try:
            result["_evaluation"] = build_evaluation(
                task,
                result,
                evaluation_workspace,
                workspace_diff,
                allow_check_command,
                artifacts,
                before_snapshot,
            )
        except Exception as exc:
            result.setdefault("warnings", []).append({"type": "evaluation", "error": str(exc)})
            result["_evaluation"] = failed_evaluation(task, result, exc, artifacts)
        finally:
            shutil.rmtree(temp_root, ignore_errors=True)
    return result


def rotate_stale_results(path: Path) -> str | None:
    if not path.exists():
        return None
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    rotated = path.with_name(f"{path.name}.previous-{stamp}")
    try:
        path.replace(rotated)
    except OSError as exc:
        raise RoleBenchError(f"cannot rotate existing results file {path}: {exc}") from exc
    return str(rotated)


def append_result(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def run_command(args: argparse.Namespace) -> int:
    config_path = Path(args.config_file).expanduser()
    roles, baseline = resolve_roles(args.roles, config_path, require_config=args.mode == "role-isolated")
    candidates, source = resolve_candidates(args)
    tasks = load_tasks(args, roles)
    validate_task_coverage(tasks, roles)
    assignments = build_assignments(args.mode, roles, baseline, candidates, args.assignment_file)
    supplied_pricing = load_pricing(args.pricing_file)
    pricing = candidate_pricing(candidates, supplied_pricing)
    judge_model = getattr(args, "judge_model", None)
    recommend = bool(getattr(args, "recommend", False))
    min_recommendation_runs = int(getattr(args, "min_recommendation_runs", 3))
    if recommend and not judge_model:
        raise RoleBenchError("--recommend requires --judge-model")
    if min_recommendation_runs < 1:
        raise RoleBenchError("--min-recommendation-runs must be >= 1")
    estimate = estimate_benchmark(roles, assignments, tasks, args.runs, pricing)
    judge_estimate = estimate_fixed_judge(roles, assignments, tasks, args.runs, judge_model, pricing)
    estimate["model_cost_usd"] = estimate["cost_usd"]
    estimate["judge"] = judge_estimate
    estimate["judge_cost_usd"] = judge_estimate["cost_usd"]
    estimate["total_cost_usd"] = (
        round((estimate["cost_usd"] or 0) + (judge_estimate["cost_usd"] or 0), 8)
        if estimate["cost_status"] == "estimated" and judge_estimate["cost_status"] in {"estimated", "not-requested"}
        else None
    )
    estimate["total_cost_status"] = "estimated" if estimate["total_cost_usd"] is not None else "unknown"
    output_dir = Path(args.output).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.jsonl"
    evaluations_path = output_dir / "evaluations.jsonl"
    judge_results_path = output_dir / "judge_results.jsonl"
    judge_evaluations_path = output_dir / "judge_evaluations.jsonl"
    recommendations_path = output_dir / "recommendations.json"
    previous_results = rotate_stale_results(results_path)
    previous_evaluations = rotate_stale_results(evaluations_path)
    previous_judge_results = rotate_stale_results(judge_results_path)
    previous_judge_evaluations = rotate_stale_results(judge_evaluations_path)
    previous_recommendations = rotate_stale_results(recommendations_path)
    command_checks = task_command_check_count(tasks)
    command_argvs = task_command_argvs(tasks)
    allow_check_command = bool(getattr(args, "allow_check_command", False))
    dimension_targets = [
        dimension for dimension in EVALUATION_DIMENSIONS
        if any(dimension in task.get("dimension_targets", []) for task in tasks)
    ]
    artifact_run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    manifest = {
        "schema_version": 2,
        "task_schema_version": 1,
        "evaluation_schema_version": 1,
        "generated_at": now_iso(),
        "omp_bin": args.omp_bin,
        "selection": source,
        "roles": roles,
        "baseline_roles": baseline,
        "mode": args.mode,
        "runs": args.runs,
        "execute": bool(args.execute),
        "executor": args.executor,
        "task_source": "file" if args.tasks_file else getattr(args, "task_source", "core"),
        "recent_work": getattr(args, "recent_work", None),
        "controller_model": args.controller_model,
        "pricing_file": args.pricing_file,
        "max_cost_usd": args.max_cost_usd,
        "previous_results": previous_results,
        "previous_evaluations": previous_evaluations,
        "previous_judge_results": previous_judge_results,
        "previous_judge_evaluations": previous_judge_evaluations,
        "previous_recommendations": previous_recommendations,
        "judge_model": judge_model,
        "judge_timeout": getattr(args, "judge_timeout", FIXED_JUDGE_TIMEOUT),
        "recommend": recommend,
        "min_recommendation_runs": min_recommendation_runs,
        "artifact_run_id": artifact_run_id,
        "evaluation": {
            "evaluator": "deterministic",
            "version": 1,
            "dimension_targets": dimension_targets,
            "hard_gate_dimensions": sorted(HARD_GATE_DIMENSIONS),
            "command_checks": command_checks,
            "command_argvs": command_argvs,
            "allow_check_command": allow_check_command,
            "model_cost_usd": 0,
            "fixed_judge": {
                "enabled": bool(judge_model),
                "model": judge_model,
                "rubric_version": FIXED_JUDGE_RUBRIC_VERSION,
                "schema_version": FIXED_JUDGE_SCHEMA_VERSION,
                "cost_usd": judge_estimate["cost_usd"],
            },
        },
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
    if args.execute and command_checks and not allow_check_command:
        summary = {"run_dir": str(output_dir), "status": "blocked", "reason": "command_checks_not_approved", "executed": False, "command_checks": command_checks}
        write_json(output_dir / "summary.json", summary)
        print("Execution blocked: task fixtures contain command checks; review argv and add --allow-check-command.", file=sys.stderr)
        return 2
    if args.execute and judge_model and judge_estimate["cost_status"] != "estimated":
        summary = {"run_dir": str(output_dir), "status": "blocked", "reason": "judge_cost_unknown", "executed": False, "estimate": estimate}
        write_json(output_dir / "summary.json", summary)
        print("Execution blocked: fixed-judge cost is unknown; provide --pricing-file with the judge rate.", file=sys.stderr)
        return 2
    if args.max_cost_usd is not None:
        if estimate["total_cost_status"] != "estimated":
            summary = {"run_dir": str(output_dir), "status": "blocked", "reason": "cost_unknown", "executed": False, "estimate": estimate}
            write_json(output_dir / "summary.json", summary)
            print("Execution blocked: model or fixed-judge cost is unknown; provide --pricing-file with all rates.", file=sys.stderr)
            return 2
        if estimate["total_cost_usd"] > args.max_cost_usd:
            summary = {"run_dir": str(output_dir), "status": "blocked", "reason": "estimated_cost_exceeds_cap", "estimated_cost_usd": estimate["total_cost_usd"], "model_cost_usd": estimate["cost_usd"], "judge_cost_usd": judge_estimate["cost_usd"], "max_cost_usd": args.max_cost_usd, "executed": False}
            write_json(output_dir / "summary.json", summary)
            print(f"Execution blocked: estimated total cost ${estimate['total_cost_usd']:.6f} exceeds cap ${args.max_cost_usd:.6f}.", file=sys.stderr)
            return 2
    if args.confirm and not args.execute:
        raise RoleBenchError("--confirm requires --execute")
    if args.execute and not args.confirm:
        summary = {"run_dir": str(output_dir), "status": "awaiting_consent", "reason": "missing_confirm", "executed": False, "estimate": estimate}
        write_json(output_dir / "summary.json", summary)
        print("Execution blocked: obtain user consent, then rerun with --execute --confirm.", file=sys.stderr)
        return 3
    result_rows = 0
    evaluation_rows = 0
    evaluation_write_errors = 0
    judge_result_rows = 0
    judge_evaluation_rows = 0
    judge_write_errors = 0
    executed_results: list[dict[str, Any]] = []
    executed_evaluations: list[dict[str, Any]] = []
    executed_judge_results: list[dict[str, Any]] = []
    executed_judge_evaluations: list[dict[str, Any]] = []
    actual_cost = 0.0
    actual_cost_seen = False
    budget_status = "not-set" if args.max_cost_usd is None else "running"
    stopped_reason = None
    if args.execute:
        artifacts_root = output_dir / "artifacts" / artifact_run_id
        ordered = list(assignments)
        random.Random(args.seed).shuffle(ordered)
        stop = False
        for assignment in ordered:
            applicable = [task for task in tasks if task_for_assignment(task, assignment)]
            for replicate in range(1, args.runs + 1):
                for task in applicable:
                    try:
                        row = run_one(
                            args.omp_bin,
                            assignment,
                            task,
                            replicate,
                            output_dir,
                            args.timeout,
                            config_path,
                            args.executor,
                            args.controller_model,
                            allow_check_command,
                            artifacts_root,
                        )
                    except Exception as exc:
                        row = {
                            "timestamp": now_iso(),
                            "assignment_id": assignment["id"],
                            "mode": assignment["mode"],
                            "role": task["role"],
                            "task_id": task["id"],
                            "replicate": replicate,
                            "model": assignment["mapping"].get(task["role"], ""),
                            "thinking": assignment.get("thinking", {}).get(task["role"]),
                            "executor": args.executor,
                            "provider_call_started": False,
                            "status": "error",
                            "execution_status": "failed",
                            "error_type": "harness_internal",
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                        row["_evaluation"] = failed_evaluation(task, row, exc)
                    evaluation = row.pop("_evaluation", None)
                    append_result(results_path, row)
                    executed_results.append(row)
                    result_rows += 1
                    if evaluation is not None:
                        try:
                            append_result(evaluations_path, evaluation)
                            executed_evaluations.append(evaluation)
                            evaluation_rows += 1
                        except (OSError, TypeError, ValueError) as exc:
                            evaluation_write_errors += 1
                            print(f"Evaluation row could not be written for {task['id']}: {exc}", file=sys.stderr)
                    cost = extract_cost(row.get("usage"))
                    if cost is not None:
                        actual_cost += cost
                        actual_cost_seen = True
                    if args.max_cost_usd is not None and row.get("provider_call_started"):
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
                    if judge_model:
                        judge_row, judge_evaluation = run_fixed_judge(
                            args.omp_bin,
                            judge_model,
                            task,
                            row,
                            evaluation or failed_evaluation(task, row, "deterministic evaluation unavailable"),
                            int(getattr(args, "judge_timeout", FIXED_JUDGE_TIMEOUT)),
                        )
                        try:
                            append_result(judge_results_path, judge_row)
                            executed_judge_results.append(judge_row)
                            judge_result_rows += 1
                        except (OSError, TypeError, ValueError) as exc:
                            judge_write_errors += 1
                            print(f"Fixed-judge row could not be written for {task['id']}: {exc}", file=sys.stderr)
                        try:
                            append_result(judge_evaluations_path, judge_evaluation)
                            executed_judge_evaluations.append(judge_evaluation)
                            judge_evaluation_rows += 1
                        except (OSError, TypeError, ValueError) as exc:
                            judge_write_errors += 1
                            print(f"Fixed-judge evaluation could not be written for {task['id']}: {exc}", file=sys.stderr)
                        judge_cost = extract_cost(judge_row.get("usage"))
                        if judge_cost is not None:
                            actual_cost += judge_cost
                            actual_cost_seen = True
                        if args.max_cost_usd is not None and judge_row.get("provider_call_started"):
                            if judge_cost is None:
                                budget_status = "unverifiable"
                                stopped_reason = "fixed_judge_did_not_return_cost"
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
    recommendations = None
    if recommend and args.execute:
        recommendations = build_recommendations(
            roles,
            tasks,
            executed_results,
            executed_evaluations,
            executed_judge_results,
            executed_judge_evaluations,
            min_recommendation_runs,
        )
        write_json(recommendations_path, recommendations)
    summary = {
        "run_dir": str(output_dir),
        "status": "completed" if args.execute else "preview",
        "assignments": len(assignments),
        "tasks": len(tasks),
        "result_rows": result_rows,
        "evaluation_rows": evaluation_rows,
        "evaluation_write_errors": evaluation_write_errors,
        "judge_result_rows": judge_result_rows,
        "judge_evaluation_rows": judge_evaluation_rows,
        "judge_write_errors": judge_write_errors,
        "recommendations": recommendations["status"] if recommendations else ("not-executed" if recommend else "not-requested"),
        "executed": bool(args.execute),
        "estimated_cost_usd": estimate["total_cost_usd"],
        "estimated_model_cost_usd": estimate["cost_usd"],
        "estimated_judge_cost_usd": judge_estimate["cost_usd"],
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
    evaluations_path = run_dir / "evaluations.jsonl"
    judge_results_path = run_dir / "judge_results.jsonl"
    judge_evaluations_path = run_dir / "judge_evaluations.jsonl"
    recommendations_path = run_dir / "recommendations.json"
    manifest = load_json_document(manifest_path)
    def load_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
        loaded = []
        if not path.exists():
            return loaded
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RoleBenchError(f"malformed {label} line {number}: {exc}") from exc
            if not isinstance(value, dict):
                raise RoleBenchError(f"malformed {label} line {number}: expected JSON object")
            loaded.append(value)
        return loaded

    rows = load_jsonl(results_path, "results.jsonl")
    evaluations = load_jsonl(evaluations_path, "evaluations.jsonl")
    judge_results = load_jsonl(judge_results_path, "judge_results.jsonl")
    judge_evaluations = load_jsonl(judge_evaluations_path, "judge_evaluations.jsonl")
    recommendations = load_json_document(recommendations_path) if recommendations_path.exists() else None
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row.get("role", ""), row.get("model", ""), row.get("thinking") or "")].append(row)
    evaluation_groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    dimension_groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for evaluation in evaluations:
        key = (evaluation.get("role", ""), evaluation.get("model", ""), evaluation.get("thinking") or "")
        evaluation_groups[key].append(evaluation)
        for dimension, value in evaluation.get("dimensions", {}).items():
            if isinstance(value, dict):
                dimension_groups[(*key, dimension)].append(value)
    lines = [
        "# OMP RoleBench Report",
        "",
        f"- Generated: {manifest.get('generated_at', 'unknown')}",
        f"- Mode: `{manifest.get('mode', 'unknown')}`",
        f"- Roles: {', '.join(manifest.get('roles', []))}",
        f"- Assignments: {len(manifest.get('assignments', []))}",
        f"- Result rows: {len(rows)}",
        f"- Evaluation rows: {len(evaluations)}",
        f"- Fixed-judge rows: {len(judge_evaluations)}",
        "",
        "## Execution",
        "",
        "| Role | Model | Thinking | Runs | Pass | Error | Pass rate | P50 latency (ms) | P95 latency (ms) | Avg output chars | Cost |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for (role, model, thinking), values in sorted(groups.items()):
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
        lines.append(f"| `{role}` | `{model}` | `{thinking or '-'}` | {len(values)} | {passes} | {errors} | {rate} | {p50_text} | {p95_text} | {avg_output} | {total_cost} |")
    if not rows:
        lines.extend(["", "No executed results found. This run contains a matrix preview only."])
    if evaluations:
        lines.extend([
            "",
            "## Outcomes",
            "",
            "| Role | Model | Thinking | Evaluations | Pass | Fail | Unknown | Hard-gate failures |",
            "|---|---|---|---:|---:|---:|---:|---:|",
        ])
        for (role, model, thinking), values in sorted(evaluation_groups.items()):
            passes = sum(value.get("outcome") == "pass" for value in values)
            failures = sum(value.get("outcome") == "fail" for value in values)
            unknown = sum(value.get("outcome") == "unknown" for value in values)
            hard_failures = sum(value.get("hard_gate_status") == "fail" for value in values)
            lines.append(f"| `{role}` | `{model}` | `{thinking or '-'}` | {len(values)} | {passes} | {failures} | {unknown} | {hard_failures} |")
        lines.extend([
            "",
            "## Dimensions",
            "",
            "| Role | Model | Thinking | Dimension | Pass | Fail | Unknown | Avg known score | Hard gate |",
            "|---|---|---|---|---:|---:|---:|---:|---|",
        ])
        for (role, model, thinking, dimension), values in sorted(dimension_groups.items()):
            passes = sum(value.get("status") == "pass" for value in values)
            failures = sum(value.get("status") == "fail" for value in values)
            unknown = sum(value.get("status") == "unknown" for value in values)
            scores = [float(value["score"]) for value in values if isinstance(value.get("score"), (int, float)) and value.get("status") != "unknown"]
            avg_score = f"{sum(scores) / len(scores):.2f}" if scores else "unknown"
            lines.append(f"| `{role}` | `{model}` | `{thinking or '-'}` | `{dimension}` | {passes} | {failures} | {unknown} | {avg_score} | {'yes' if dimension in HARD_GATE_DIMENSIONS else 'no'} |")
        failed_checks = [
            (evaluation, check)
            for evaluation in evaluations
            for check in evaluation.get("checks", [])
            if isinstance(check, dict) and check.get("status") == "fail"
        ]
        if failed_checks:
            lines.extend(["", "## Failed Checks", ""])
            for evaluation, check in failed_checks:
                evidence = evidence_text(check.get("evidence", "")).replace("|", "\\|")
                lines.append(f"- `{evaluation.get('task_id', '')}` / `{evaluation.get('model', '')}` / `{check.get('id', '')}`: {evidence}")
    if judge_evaluations:
        lines.extend([
            "",
            "## Fixed Judge",
            "",
            f"Rubric: `{judge_evaluations[0].get('rubric_version', 'unknown')}`",
            "",
            "| Role | Task | Model | Outcome | Hard gate | Summary |",
            "|---|---|---|---|---|---|",
        ])
        for evaluation in judge_evaluations:
            summary = evidence_text(evaluation.get("summary") or "").replace("|", "\\|")
            lines.append(f"| `{evaluation.get('role', '')}` | `{evaluation.get('task_id', '')}` | `{evaluation.get('model', '')}` | {evaluation.get('outcome', 'unknown')} | {evaluation.get('hard_gate_status', 'unknown')} | {summary} |")
    if isinstance(recommendations, dict):
        def recommendation_links(item: dict[str, Any]) -> str:
            paths = item.get("evidence", [])
            if not isinstance(paths, list):
                return ""
            return ", ".join(f"[{path}]({path})" for path in paths if isinstance(path, str))

        lines.extend(["", "## Recommendations", "", f"Status: `{recommendations.get('status', 'unknown')}`", "", "| Role | Model | Thinking | Status | Reason | Evidence |", "|---|---|---|---|---|---|"])
        for item in recommendations.get("roles", []):
            candidate = item.get("candidate") if isinstance(item.get("candidate"), dict) else {}
            reason = evidence_text(item.get("reason", "")).replace("|", "\\|")
            lines.append(f"| `{item.get('role', '')}` | `{item.get('model') or '-'}` | `{item.get('thinking') or '-'}` | {item.get('status', 'blocked')} | {reason} | {recommendation_links(candidate)} |")
        candidates = recommendations.get("candidates", [])
        if isinstance(candidates, list) and candidates:
            lines.extend(["", "### Recommendation Candidates", "", "| Role | Model | Thinking | Eligible | Hard dimensions | Quality | Blocked reasons | Evidence |", "|---|---|---|---|---|---:|---|---|"])
            for candidate in candidates:
                if not isinstance(candidate, dict):
                    continue
                reasons = evidence_text("; ".join(str(reason) for reason in candidate.get("blocked_reasons", []))).replace("|", "\\|")
                hard_dimensions = ", ".join(candidate.get("judge_hard_dimensions", [])) or "none"
                quality = candidate.get("quality_score") if candidate.get("quality_score") is not None else "unknown"
                lines.append(f"| `{candidate.get('role', '')}` | `{candidate.get('model', '')}` | `{candidate.get('thinking') or '-'}` | {candidate.get('eligible', False)} | `{hard_dimensions}` | {quality} | {reasons or '-'} | {recommendation_links(candidate)} |")
    report = "\n".join(lines) + "\n"
    if args.format == "json":
        serializable_groups = {
            f"{role}/{model}:{thinking}" if thinking else f"{role}/{model}": values
            for (role, model, thinking), values in groups.items()
        }
        serializable_evaluation_groups = {
            f"{role}/{model}:{thinking}" if thinking else f"{role}/{model}": values
            for (role, model, thinking), values in evaluation_groups.items()
        }
        rendered = json.dumps({"manifest": manifest, "groups": serializable_groups, "evaluations": evaluations, "evaluation_groups": serializable_evaluation_groups, "judge_results": judge_results, "judge_evaluations": judge_evaluations, "recommendations": recommendations}, ensure_ascii=False, indent=2) + "\n"
        if args.output:
            output = Path(args.output).expanduser()
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(rendered, encoding="utf-8")
            print(output)
        else:
            print(rendered, end="")
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
    run.add_argument("--task-source", choices=("core", "recent"), default="core", help="task source when --tasks-file is not supplied")
    run.add_argument("--recent-workspace", action="append", default=None, help="workspace to inspect for --task-source recent; repeat or use comma-separated paths")
    run.add_argument("--recent-days", type=int, default=14, help="lookback window for recent work")
    run.add_argument("--recent-commits", type=int, default=5, help="maximum recent commits to include")
    run.add_argument("--runs", type=int, default=1)
    run.add_argument("--execute", action="store_true", help="invoke OMP; otherwise only write manifest")
    run.add_argument("--executor", choices=("direct", "delegated"), default="direct", help="direct model call or a temporary OMP task agent")
    run.add_argument("--controller-model", help="controller model used by delegated executor")
    run.add_argument("--confirm", action="store_true", help="confirm the printed plan and cost estimate before execution")
    run.add_argument("--allow-check-command", action="store_true", help="allow reviewed argv-based command checks from task fixtures")
    run.add_argument("--pricing-file", help="JSON rates keyed by provider/model, in USD per million input/output tokens")
    run.add_argument("--max-cost-usd", "--budget-usd", dest="max_cost_usd", type=float, help="maximum allowed estimated/observed benchmark cost in USD")
    run.add_argument("--judge-model", help="fixed semantic judge provider/model selector; adds a separately budgeted judge call per cell")
    run.add_argument("--judge-timeout", type=int, default=FIXED_JUDGE_TIMEOUT, help="fixed judge timeout in seconds")
    run.add_argument("--recommend", action="store_true", help="write recommendations.json after fixed-judge evaluation")
    run.add_argument("--min-recommendation-runs", type=int, default=3, help="minimum fixed runs per task before recommending a model")
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
            if args.recent_days < 1:
                raise RoleBenchError("--recent-days must be >= 1")
            if args.recent_commits < 1:
                raise RoleBenchError("--recent-commits must be >= 1")
            if args.max_cost_usd is not None and args.max_cost_usd < 0:
                raise RoleBenchError("--max-cost-usd must be >= 0")
            if args.judge_timeout < 1:
                raise RoleBenchError("--judge-timeout must be >= 1")
            if args.min_recommendation_runs < 1:
                raise RoleBenchError("--min-recommendation-runs must be >= 1")
            return run_command(args)
        if args.command == "report":
            return report_command(args)
        raise RoleBenchError(f"unknown command: {args.command}")
    except RoleBenchError as exc:
        print(f"rolebench: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
