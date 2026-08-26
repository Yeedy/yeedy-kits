#!/usr/bin/env python3
"""Collect read-only memory health metrics for selected agent runtimes."""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import closing
import json
import os
import re
import shutil
import sqlite3
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import quote


SKILL_NAME = "optimize-agent-memory"
OMP_PROFILE_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
OMP_RESERVED_PROFILE_PATTERN = re.compile(
    r"^(?:con|prn|aux|nul|com[0-9]|lpt[0-9])(?:\.|$)", re.IGNORECASE
)
OMP_ARTIFACT_NAMES = (
    "MEMORY.md",
    "memory_summary.md",
    "learned.md",
    "raw_memories.md",
)
CODEX_ARTIFACT_NAMES = (
    "MEMORY.md",
    "memory_summary.md",
    "raw_memories.md",
)


def file_stats(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"exists": False, "bytes": 0, "lines": 0}
    data = path.read_bytes()
    return {
        "exists": True,
        "bytes": len(data),
        "lines": data.count(b"\n") + (1 if data and not data.endswith(b"\n") else 0),
    }


def read_jobs(db_path: Path) -> dict[str, Any]:
    empty_state = {
        "database": str(db_path),
        "exists": db_path.is_file(),
        "job_count": 0,
        "status_counts": {},
        "attention_jobs": [],
    }
    if not db_path.is_file():
        return empty_state

    uri = f"file:{quote(str(db_path), safe='/')}?mode=ro"
    try:
        return _read_jobs_from_database(uri, db_path, uri=True)
    except sqlite3.OperationalError:
        with tempfile.TemporaryDirectory(prefix="memory-baseline-") as directory:
            temporary_db = Path(directory) / "memory.sqlite"
            shutil.copy2(db_path, temporary_db)
            for suffix in ("-wal", "-shm"):
                sidecar = Path(f"{db_path}{suffix}")
                if sidecar.is_file():
                    shutil.copy2(sidecar, Path(f"{temporary_db}{suffix}"))
            return _read_jobs_from_database(str(temporary_db), db_path)


def _read_jobs_from_database(
    target: str, report_path: Path, *, uri: bool = False
) -> dict[str, Any]:
    with closing(sqlite3.connect(target, uri=uri)) as db:
        return _read_job_rows(db, report_path)


def _read_job_rows(db: sqlite3.Connection, db_path: Path) -> dict[str, Any]:
    empty_state = {
        "database": str(db_path),
        "exists": True,
        "job_count": 0,
        "status_counts": {},
        "attention_jobs": [],
    }
    table = db.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'jobs'"
    ).fetchone()
    if table is None:
        return empty_state
    rows = db.execute(
        """
        SELECT kind, job_key, status, retry_remaining, last_error,
               input_watermark, last_success_watermark
        FROM jobs
        ORDER BY kind, job_key
        """
    ).fetchall()

    jobs = [
        {
            "kind": row[0],
            "job_key": row[1],
            "status": row[2],
            "retry_remaining": row[3],
            "last_error": row[4],
            "input_watermark": row[5],
            "last_success_watermark": row[6],
        }
        for row in rows
    ]
    status_counts = Counter(job["status"] for job in jobs)
    attention_jobs = [
        job for job in jobs if job["status"] != "done" or job["last_error"] is not None
    ]
    return {
        "database": str(db_path),
        "exists": True,
        "job_count": len(jobs),
        "status_counts": dict(sorted(status_counts.items())),
        "attention_jobs": attention_jobs,
    }


def scope_stats(root: Path, artifact_names: tuple[str, ...]) -> dict[str, Any]:
    artifacts = {name: file_stats(root / name) for name in artifact_names}
    rollouts_dir = root / "rollout_summaries"
    rollout_count = (
        sum(1 for path in rollouts_dir.iterdir() if path.is_file() and path.suffix == ".md")
        if rollouts_dir.is_dir()
        else 0
    )
    return {
        "root": str(root),
        "generated_artifacts_available": (
            artifacts["MEMORY.md"]["exists"] and artifacts["memory_summary.md"]["exists"]
        ),
        "artifacts": artifacts,
        "rollout_summary_count": rollout_count,
    }


def collect_omp(agent_dir: Path) -> dict[str, Any]:
    memories_dir = agent_dir / "memories"
    if not agent_dir.is_dir():
        raise FileNotFoundError(f"OMP agent directory not found: {agent_dir}")
    scopes = (
        [
            scope_stats(path, OMP_ARTIFACT_NAMES)
            for path in sorted(memories_dir.iterdir())
            if path.is_dir()
        ]
        if memories_dir.is_dir()
        else []
    )
    return {
        "runtime": "omp",
        "agent_dir": str(agent_dir),
        "scope_count": len(scopes),
        "scopes": scopes,
        "job_state": read_jobs(agent_dir / "agent.db"),
    }


def resolve_codex_db(codex_dir: Path, requested_db: Path | None) -> Path:
    if requested_db is not None:
        db_path = requested_db.expanduser().resolve()
        if not db_path.is_file():
            raise FileNotFoundError(f"Codex memory database not found: {db_path}")
        return db_path

    candidates = (
        codex_dir / "memories_1.sqlite",
        codex_dir / "sqlite" / "memories_1.sqlite",
    )
    existing = [path for path in candidates if path.is_file()]
    if len(existing) > 1:
        paths = ", ".join(str(path) for path in existing)
        raise RuntimeError(
            "Multiple Codex memory databases found; select one with --codex-db: "
            f"{paths}"
        )
    return existing[0] if existing else candidates[0]


def collect_codex(codex_dir: Path, requested_db: Path | None = None) -> dict[str, Any]:
    memories_dir = codex_dir / "memories"
    if not codex_dir.is_dir():
        raise FileNotFoundError(f"Codex directory not found: {codex_dir}")
    if not memories_dir.is_dir():
        raise FileNotFoundError(f"Codex memories directory not found: {memories_dir}")
    db_path = resolve_codex_db(codex_dir, requested_db)
    result = scope_stats(memories_dir, CODEX_ARTIFACT_NAMES)
    result.update(
        {
            "runtime": "codex",
            "codex_dir": str(codex_dir),
            "ad_hoc_note_count": len(
                list((memories_dir / "extensions" / "ad_hoc" / "notes").glob("*.md"))
            ),
            "git_repository": (memories_dir / ".git").is_dir(),
            "job_state": read_jobs(db_path),
        }
    )
    return result


def resolve_codex_dir(home: Path) -> Path:
    configured_codex_dir = os.environ.get("CODEX_HOME")
    codex_dir = Path(configured_codex_dir) if configured_codex_dir else home / ".codex"
    return codex_dir.expanduser().resolve()


def resolve_omp_agent_dir(home: Path) -> Path:
    configured_agent_dir = os.environ.get("PI_CODING_AGENT_DIR")
    if configured_agent_dir:
        return Path(configured_agent_dir).expanduser().resolve()

    configured_config_dir = os.environ.get("PI_CONFIG_DIR")
    config_dir = Path(configured_config_dir or ".omp").expanduser()
    config_root = config_dir if config_dir.is_absolute() else home / config_dir
    raw_profile = (
        os.environ["OMP_PROFILE"]
        if "OMP_PROFILE" in os.environ
        else os.environ.get("PI_PROFILE")
    )
    profile = raw_profile.strip() if raw_profile is not None else None
    if profile and profile != "default":
        if (
            OMP_PROFILE_PATTERN.fullmatch(profile) is None
            or profile.endswith(".")
            or OMP_RESERVED_PROFILE_PATTERN.match(profile) is not None
        ):
            raise ValueError(f"Invalid OMP profile name: {profile}")
        return (config_root / "profiles" / profile / "agent").resolve()
    return (config_root / "agent").resolve()


def detect_installed_runtime(
    script_path: Path, codex_dir: Path, omp_agent_dir: Path
) -> str | None:
    install_roots = {
        "codex": codex_dir / "skills" / SKILL_NAME,
        "omp": omp_agent_dir / "skills" / SKILL_NAME,
    }
    resolved_script = script_path.resolve()
    matches = [
        runtime
        for runtime, root in install_roots.items()
        if root.resolve() in resolved_script.parents
    ]
    if len(matches) > 1:
        raise RuntimeError(f"Ambiguous installed runtime for script: {script_path}")
    return matches[0] if matches else None


def parse_args() -> argparse.Namespace:
    home = Path.home()
    codex_dir = resolve_codex_dir(home)
    omp_agent_dir = resolve_omp_agent_dir(home)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", choices=("omp", "codex", "both"))
    parser.add_argument("--omp-dir", type=Path, default=omp_agent_dir)
    parser.add_argument("--codex-dir", type=Path, default=codex_dir)
    parser.add_argument("--codex-db", type=Path)
    args = parser.parse_args()

    installed_runtime = detect_installed_runtime(
        Path(__file__), codex_dir, omp_agent_dir
    )
    if args.runtime is None and installed_runtime is not None:
        args.runtime = installed_runtime
    if args.runtime is None:
        parser.error("--runtime is required outside a recognized Codex or OMP installation")
    return args


def main() -> None:
    args = parse_args()
    results = []
    if args.runtime in ("omp", "both"):
        results.append(collect_omp(args.omp_dir.expanduser().resolve()))
    if args.runtime in ("codex", "both"):
        results.append(
            collect_codex(args.codex_dir.expanduser().resolve(), args.codex_db)
        )
    print(json.dumps(results[0] if len(results) == 1 else results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
