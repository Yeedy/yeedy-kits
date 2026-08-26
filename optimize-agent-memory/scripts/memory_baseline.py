#!/usr/bin/env python3
"""Collect read-only OMP and Codex memory health metrics."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import sqlite3
from pathlib import Path
from typing import Any


ARTIFACT_NAMES = (
    "MEMORY.md",
    "memory_summary.md",
    "learned.md",
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
    if not db_path.is_file():
        return {"database": str(db_path), "exists": False, "jobs": []}

    uri = f"file:{db_path}?mode=ro"
    with sqlite3.connect(uri, uri=True) as db:
        table = db.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'jobs'"
        ).fetchone()
        if table is None:
            return {"database": str(db_path), "exists": True, "jobs": []}
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


def scope_stats(root: Path) -> dict[str, Any]:
    artifacts = {name: file_stats(root / name) for name in ARTIFACT_NAMES}
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
        [scope_stats(path) for path in sorted(memories_dir.iterdir()) if path.is_dir()]
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


def collect_codex(codex_dir: Path) -> dict[str, Any]:
    memories_dir = codex_dir / "memories"
    if not codex_dir.is_dir():
        raise FileNotFoundError(f"Codex directory not found: {codex_dir}")
    if not memories_dir.is_dir():
        raise FileNotFoundError(f"Codex memories directory not found: {memories_dir}")
    db_path = codex_dir / "memories_1.sqlite"
    if not db_path.is_file():
        db_path = codex_dir / "sqlite" / "memories_1.sqlite"
    result = scope_stats(memories_dir)
    result.update(
        {
            "runtime": "codex",
            "codex_dir": str(codex_dir),
            "ad_hoc_note_count": len(list((memories_dir / "extensions" / "ad_hoc" / "notes").glob("*.md"))),
            "git_repository": (memories_dir / ".git").is_dir(),
            "job_state": read_jobs(db_path),
        }
    )
    return result


def parse_args() -> argparse.Namespace:
    home = Path.home()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", choices=("omp", "codex", "both"), required=True)
    parser.add_argument("--omp-dir", type=Path, default=home / ".omp" / "agent")
    parser.add_argument("--codex-dir", type=Path, default=home / ".codex")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = []
    if args.runtime in ("omp", "both"):
        results.append(collect_omp(args.omp_dir.expanduser().resolve()))
    if args.runtime in ("codex", "both"):
        results.append(collect_codex(args.codex_dir.expanduser().resolve()))
    print(json.dumps(results[0] if len(results) == 1 else results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
