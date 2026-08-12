#!/usr/bin/env python3
"""Initialize and update a workspace-local learning profile."""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import tempfile
from collections import Counter
from pathlib import Path


OUTPUT_RELATIVE_PATH = Path(".ai-coding-daily-coach/project-scope.md")
PROFILE_RELATIVE_PATH = Path(".ai-coding-daily-coach/profile.json")
IGNORED_DIRS = {
    ".ai-coding-daily-coach",
    ".git",
    ".gradle",
    ".idea",
    ".next",
    ".nuxt",
    ".venv",
    ".vscode",
    "build",
    "coverage",
    "dist",
    "node_modules",
    "target",
    "vendor",
}
LANGUAGES = {
    ".c": "C",
    ".cpp": "C++",
    ".cs": "C#",
    ".go": "Go",
    ".java": "Java",
    ".js": "JavaScript",
    ".jsx": "JavaScript",
    ".kt": "Kotlin",
    ".php": "PHP",
    ".py": "Python",
    ".rb": "Ruby",
    ".rs": "Rust",
    ".scala": "Scala",
    ".swift": "Swift",
    ".ts": "TypeScript",
    ".tsx": "TypeScript",
    ".vue": "Vue",
}
MANIFEST_LABELS = {
    "Cargo.toml": "Cargo",
    "build.gradle": "Gradle",
    "build.gradle.kts": "Gradle Kotlin DSL",
    "composer.json": "Composer",
    "go.mod": "Go modules",
    "package.json": "Node.js",
    "pom.xml": "Maven",
    "pyproject.toml": "Python",
    "requirements.txt": "Python requirements",
}


def walk_workspace(root: Path, max_depth: int = 6):
    for current, dirs, files in os.walk(root):
        relative = Path(current).relative_to(root)
        depth = len(relative.parts)
        if depth >= max_depth:
            dirs[:] = []
        else:
            dirs[:] = sorted(item for item in dirs if item not in IGNORED_DIRS)
        if depth >= max_depth:
            continue
        for name in sorted(files):
            yield Path(current) / name


def package_details(path: Path) -> tuple[str | None, list[str]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None, []
    if not isinstance(data, dict):
        return None, []
    dependencies = set()
    for field in ("dependencies", "devDependencies", "peerDependencies"):
        values = data.get(field, {})
        if isinstance(values, dict):
            dependencies.update(str(key) for key in values)
    return data.get("name") if isinstance(data.get("name"), str) else None, sorted(dependencies)


def codex_home() -> Path:
    configured = os.environ.get("CODEX_HOME")
    return Path(configured).expanduser().resolve() if configured else Path.home() / ".codex"


def has_project_evidence(root: Path) -> bool:
    if root == Path.home() or root.parent == Path.home():
        return False
    for path in walk_workspace(root, max_depth=4):
        if path.name in MANIFEST_LABELS or path.suffix.lower() in LANGUAGES:
            return True
    return False


def remove_nested_roots(paths: set[Path]) -> list[Path]:
    selected: list[Path] = []
    for path in sorted(paths, key=lambda item: (len(item.parts), str(item))):
        if any(path.is_relative_to(parent) for parent in selected):
            continue
        selected.append(path)
    return sorted(selected)


def discover_codex_workspaces() -> list[Path]:
    home = codex_home()
    databases = sorted({*home.glob("state_*.sqlite"), *(home / "sqlite").glob("state_*.sqlite")})
    if not databases:
        raise SystemExit(f"cannot discover Codex workspaces: no state database under {home}")

    found: set[Path] = set()
    usable_database = False
    for database in databases:
        try:
            connection = sqlite3.connect(f"file:{database}?mode=ro", uri=True)
            columns = {row[1] for row in connection.execute("PRAGMA table_info(threads)")}
            if "cwd" not in columns:
                connection.close()
                continue
            usable_database = True
            for (value,) in connection.execute("SELECT DISTINCT cwd FROM threads WHERE cwd != ''"):
                path = Path(value).expanduser().resolve()
                if path.is_dir() and has_project_evidence(path):
                    found.add(path)
            connection.close()
        except sqlite3.Error as exc:
            raise SystemExit(f"cannot read Codex workspace database {database}: {exc}") from exc
    if not usable_database:
        raise SystemExit("cannot discover Codex workspaces: threads.cwd is unavailable")
    if not found:
        raise SystemExit("cannot discover Codex workspaces: no accessible workspace paths found")
    return remove_nested_roots(found)


def validate_roots(values: list[str]) -> list[Path]:
    roots = []
    for value in values:
        path = Path(value).expanduser().resolve()
        if not path.is_dir():
            raise SystemExit(f"scan root is not a directory: {path}")
        if path not in roots:
            roots.append(path)
    return roots


def scan(roots: list[Path]) -> dict:
    languages: Counter[str] = Counter()
    manifests: list[tuple[str, Path, str]] = []
    package_names: list[str] = []
    dependencies: Counter[str] = Counter()

    for root in roots:
        for path in walk_workspace(root):
            language = LANGUAGES.get(path.suffix.lower())
            if language:
                languages[language] += 1
            label = MANIFEST_LABELS.get(path.name)
            if label:
                manifests.append((root.name, path.relative_to(root), label))
            if path.name == "package.json":
                name, found = package_details(path)
                if name:
                    package_names.append(name)
                dependencies.update(found)

    return {
        "languages": languages,
        "manifests": manifests,
        "package_names": package_names,
        "dependencies": dependencies,
    }


def render(roots: list[Path], fallback: Path, focus_areas: list[str]) -> str:
    evidence = scan(roots)
    languages = evidence["languages"]
    manifests = evidence["manifests"]
    package_names = evidence["package_names"]
    dependencies = evidence["dependencies"]

    if not languages and not manifests and not focus_areas:
        try:
            return fallback.read_text(encoding="utf-8").rstrip() + "\n"
        except OSError as exc:
            raise SystemExit(f"cannot read fallback profile {fallback}: {exc}") from exc

    lines = [
        "# Current Learning Profile",
        "",
        "Generated from the user's development focus and project evidence in the current workspace. Use it only as scenario context; do not test repository trivia.",
        "",
    ]
    if focus_areas:
        lines.extend(["## Development Focus", ""])
        lines.extend(f"- {item}" for item in focus_areas)
        lines.append("")
    if manifests:
        lines.extend(["## Project Signals", ""])
        for workspace, path, label in manifests[:30]:
            lines.append(f"- `{workspace}/{path}`: {label}")
        lines.append("")
    if languages:
        lines.extend(["## Dominant Languages", ""])
        for language, count in languages.most_common(8):
            lines.append(f"- {language}: {count} source files")
        lines.append("")
    if package_names or dependencies:
        lines.extend(["## Node.js Context", ""])
        if package_names:
            lines.append("- Packages: " + ", ".join(f"`{name}`" for name in sorted(set(package_names))[:12]))
        if dependencies:
            lines.append("- Declared dependencies: " + ", ".join(f"`{name}`" for name, _ in dependencies.most_common(30)))
        lines.append("")
    lines.extend(
        [
            "## Exercise Guidance",
            "",
            "- Ground scenarios in the detected stack without requiring repository-specific knowledge.",
            "- Prefer transferable judgment about agent workflows, boundaries, verification, and recovery.",
            "- Do not expose file contents, credentials, or confidential business details.",
            "",
        ]
    )
    return "\n".join(lines)


def save(path: Path, content: str) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
            handle.write(content)
            temporary = Path(handle.name)
        os.replace(temporary, path)
    except OSError as exc:
        raise SystemExit(f"cannot write workspace profile {path}: {exc}") from exc


def load_profile(path: Path) -> dict:
    try:
        profile = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SystemExit(f"cannot read valid profile {path}: {exc}") from exc
    if not isinstance(profile, dict) or profile.get("version") != 1:
        raise SystemExit(f"unsupported profile schema in {path}")
    focus_areas = profile.get("focus_areas")
    if not isinstance(focus_areas, list) or not all(isinstance(item, str) for item in focus_areas):
        raise SystemExit(f"invalid focus_areas in {path}")
    scan_mode = profile.get("scan_mode")
    scan_roots = profile.get("scan_roots")
    if scan_mode not in {"all_codex_workspaces", "selected"}:
        raise SystemExit(f"invalid scan_mode in {path}")
    if not isinstance(scan_roots, list) or not all(isinstance(item, str) for item in scan_roots):
        raise SystemExit(f"invalid scan_roots in {path}")
    return profile


def normalize_focus(values: list[str]) -> list[str]:
    result = []
    for value in values:
        normalized = value.strip()
        if normalized and normalized not in result:
            result.append(normalized)
    return result


def write_profile(
    root: Path,
    fallback: Path,
    focus_areas: list[str],
    scan_mode: str,
    scan_roots: list[Path],
) -> None:
    profile_path = root / PROFILE_RELATIVE_PATH
    output_path = root / OUTPUT_RELATIVE_PATH
    rendered_scope = render(scan_roots, fallback, focus_areas)
    profile_content = json.dumps(
        {
            "version": 1,
            "focus_areas": focus_areas,
            "scan_mode": scan_mode,
            "scan_roots": [str(path) for path in scan_roots],
        },
        ensure_ascii=False,
        indent=2,
    ) + "\n"
    save(
        profile_path,
        profile_content,
    )
    save(output_path, rendered_scope)
    print(
        json.dumps(
            {
                "profile_path": str(profile_path),
                "scope_path": str(output_path),
                "focus_areas": focus_areas,
                "scan_mode": scan_mode,
                "scan_roots": [str(path) for path in scan_roots],
            },
            ensure_ascii=False,
        )
    )


def cmd_init(args: argparse.Namespace) -> None:
    profile_path = args.root / PROFILE_RELATIVE_PATH
    if profile_path.exists():
        raise SystemExit(f"profile already initialized: {profile_path}; use update")
    scan_mode = "selected" if args.scan_root else "all_codex_workspaces"
    scan_roots = validate_roots(args.scan_root) if args.scan_root else discover_codex_workspaces()
    write_profile(args.root, args.fallback, normalize_focus(args.focus), scan_mode, scan_roots)


def cmd_update(args: argparse.Namespace) -> None:
    profile_path = args.root / PROFILE_RELATIVE_PATH
    if not profile_path.exists():
        raise SystemExit(f"profile is not initialized: {profile_path}; use init")
    profile = load_profile(profile_path)
    focus_areas = normalize_focus(args.focus) if args.focus is not None else profile["focus_areas"]
    if args.scan_root:
        scan_mode = "selected"
        scan_roots = validate_roots(args.scan_root)
    elif args.all_workspaces or profile["scan_mode"] == "all_codex_workspaces":
        scan_mode = "all_codex_workspaces"
        scan_roots = discover_codex_workspaces()
    else:
        scan_mode = "selected"
        scan_roots = validate_roots(profile["scan_roots"])
    write_profile(args.root, args.fallback, focus_areas, scan_mode, scan_roots)


def cmd_show(args: argparse.Namespace) -> None:
    profile_path = args.root / PROFILE_RELATIVE_PATH
    profile = load_profile(profile_path)
    profile["profile_path"] = str(profile_path)
    profile["scope_path"] = str(args.root / OUTPUT_RELATIVE_PATH)
    print(json.dumps(profile, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=".", type=lambda value: Path(value).resolve(), help="workspace root; defaults to cwd")
    parser.add_argument("--fallback", type=lambda value: Path(value).resolve(), help="generic fallback Markdown file")
    sub = parser.add_subparsers(dest="command", required=True)

    init = sub.add_parser("init")
    init.add_argument("--focus", action="append", default=[], help="development direction; repeat as needed")
    init.add_argument("--scan-root", action="append", default=[], help="workspace to scan instead of all Codex workspaces; repeat as needed")
    init.set_defaults(func=cmd_init)

    update = sub.add_parser("update")
    update.add_argument("--focus", action="append", default=None, help="replace development directions; repeat as needed")
    update_scope = update.add_mutually_exclusive_group()
    update_scope.add_argument("--scan-root", action="append", default=[], help="replace selected scan roots; repeat as needed")
    update_scope.add_argument("--all-workspaces", action="store_true", help="switch back to all discoverable Codex workspaces")
    update.set_defaults(func=cmd_update)

    sub.add_parser("show").set_defaults(func=cmd_show)
    args = parser.parse_args()

    if not args.root.is_dir():
        raise SystemExit(f"workspace root is not a directory: {args.root}")
    if args.command in {"init", "update"} and args.fallback is None:
        parser.error("--fallback is required for init and update")
    args.func(args)


if __name__ == "__main__":
    main()
