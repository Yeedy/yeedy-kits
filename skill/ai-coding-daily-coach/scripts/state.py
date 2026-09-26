#!/usr/bin/env python3
"""Manage workspace-local learning state for ai-coding-daily-coach."""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from datetime import date, timedelta
from pathlib import Path


STATE_RELATIVE_PATH = Path(".ai-coding-daily-coach/state.json")
INTERVALS = (1, 3, 7, 14, 30)


def empty_state() -> dict:
    return {
        "version": 1,
        "pending_exercise": None,
        "pending_review": None,
        "history": [],
        "reviews": [],
    }


def parse_day(value: str | None) -> date:
    try:
        return date.fromisoformat(value) if value else date.today()
    except ValueError as exc:
        raise SystemExit(f"invalid ISO date: {value}") from exc


def state_path(root: str) -> Path:
    return Path(root).resolve() / STATE_RELATIVE_PATH


def load(path: Path) -> dict:
    if not path.exists():
        return empty_state()
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"cannot read valid state from {path}: {exc}") from exc
    required = set(empty_state())
    if not isinstance(state, dict) or not required.issubset(state):
        raise SystemExit(f"invalid state schema in {path}")
    return state


def save(path: Path, state: dict) -> None:
    payload = json.dumps(state, ensure_ascii=False, indent=2) + "\n"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent, delete=False
        ) as handle:
            handle.write(payload)
            temporary = Path(handle.name)
        os.replace(temporary, path)
    except OSError as exc:
        raise SystemExit(f"cannot write state to {path}: {exc}") from exc


def find_review(state: dict, concept: str) -> dict | None:
    return next((item for item in state["reviews"] if item["concept"] == concept), None)


def cmd_show(args: argparse.Namespace) -> None:
    path = state_path(args.root)
    state = load(path)
    state["state_path"] = str(path)
    print(json.dumps(state, ensure_ascii=False, indent=2))


def cmd_init(args: argparse.Namespace) -> None:
    path = state_path(args.root)
    state = load(path)
    if not path.exists():
        save(path, state)
    print(json.dumps({"state_path": str(path), "initialized": True}, ensure_ascii=False))


def cmd_question(args: argparse.Namespace) -> None:
    path = state_path(args.root)
    state = load(path)
    if state["pending_exercise"] is not None:
        raise SystemExit("a pending exercise already exists")
    asked = parse_day(args.date).isoformat()
    state["pending_exercise"] = {
        "id": f"{asked}-{len(state['history']) + 1}",
        "asked_at": asked,
        "topic": args.topic,
        "difficulty": args.difficulty,
        "concept_question": args.concept_question,
        "scenario_question": args.scenario_question,
        "sources": args.source,
        "answered_parts": [],
    }
    save(path, state)
    print(json.dumps(state["pending_exercise"], ensure_ascii=False, indent=2))


def cmd_partial(args: argparse.Namespace) -> None:
    path = state_path(args.root)
    state = load(path)
    pending = state["pending_exercise"]
    if pending is None:
        raise SystemExit("no pending exercise")
    if args.part not in pending["answered_parts"]:
        pending["answered_parts"].append(args.part)
        pending["answered_parts"].sort()
    save(path, state)
    print(json.dumps(pending, ensure_ascii=False, indent=2))


def validate_score(name: str, value: float, maximum: float) -> None:
    if not math.isfinite(value) or value < 0 or value > maximum:
        raise SystemExit(f"{name} must be between 0 and {maximum}")


def cmd_grade(args: argparse.Namespace) -> None:
    path = state_path(args.root)
    state = load(path)
    pending = state["pending_exercise"]
    if pending is None:
        raise SystemExit("no pending exercise")
    missing = {"concept", "scenario"} - set(pending["answered_parts"])
    if missing:
        raise SystemExit(f"cannot grade incomplete exercise; missing: {', '.join(sorted(missing))}")

    validate_score("concept score", args.concept_score, 4)
    validate_score("scenario score", args.scenario_score, 4)
    validate_score("explanation score", args.explanation_score, 2)
    graded = parse_day(args.date)
    total = round(args.concept_score + args.scenario_score + args.explanation_score, 1)
    entry = {
        **pending,
        "graded_at": graded.isoformat(),
        "score": total,
        "breakdown": {
            "concept": args.concept_score,
            "scenario": args.scenario_score,
            "explanation": args.explanation_score,
        },
        "weak_concepts": args.weak_concept,
    }
    state["history"].append(entry)

    for concept in args.weak_concept:
        review = find_review(state, concept)
        if review is None:
            review = {"concept": concept}
            state["reviews"].append(review)
        review.update(
            {
                "source_exercise_id": pending["id"],
                "source_topic": pending["topic"],
                "interval_index": 0,
                "last_result": "new",
                "last_reviewed": None,
                "next_due": (graded + timedelta(days=INTERVALS[0])).isoformat(),
            }
        )

    state["pending_exercise"] = None
    save(path, state)
    print(json.dumps(entry, ensure_ascii=False, indent=2))


def cmd_due(args: argparse.Namespace) -> None:
    state = load(state_path(args.root))
    today = parse_day(args.date).isoformat()
    due = sorted(
        (item for item in state["reviews"] if item["next_due"] <= today),
        key=lambda item: item["next_due"],
    )
    print(json.dumps(due, ensure_ascii=False, indent=2))


def cmd_ask_review(args: argparse.Namespace) -> None:
    path = state_path(args.root)
    state = load(path)
    if state["pending_review"] is not None:
        raise SystemExit("a pending review already exists")
    review = find_review(state, args.concept)
    if review is None:
        raise SystemExit(f"unknown review concept: {args.concept}")
    state["pending_review"] = {
        "concept": args.concept,
        "question": args.question,
        "asked_at": parse_day(args.date).isoformat(),
        "source_topic": review["source_topic"],
    }
    save(path, state)
    print(json.dumps(state["pending_review"], ensure_ascii=False, indent=2))


def cmd_review(args: argparse.Namespace) -> None:
    path = state_path(args.root)
    state = load(path)
    pending = state["pending_review"]
    if pending is None:
        raise SystemExit("no pending review")
    review = find_review(state, pending["concept"])
    if review is None:
        raise SystemExit(f"missing review record: {pending['concept']}")

    reviewed = parse_day(args.date)
    if args.result == "pass":
        review["interval_index"] = min(review["interval_index"] + 1, len(INTERVALS) - 1)
    else:
        review["interval_index"] = 0
    review["last_result"] = args.result
    review["last_reviewed"] = reviewed.isoformat()
    review["next_due"] = (
        reviewed + timedelta(days=INTERVALS[review["interval_index"]])
    ).isoformat()
    state["pending_review"] = None
    save(path, state)
    print(json.dumps(review, ensure_ascii=False, indent=2))


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--root", default=".", help="workspace root; defaults to cwd")
    sub = result.add_subparsers(dest="command", required=True)

    sub.add_parser("show").set_defaults(func=cmd_show)
    sub.add_parser("init").set_defaults(func=cmd_init)

    question = sub.add_parser("question")
    question.add_argument("--topic", required=True)
    question.add_argument("--difficulty", required=True, choices=("基础", "进阶", "挑战"))
    question.add_argument("--concept-question", required=True)
    question.add_argument("--scenario-question", required=True)
    question.add_argument("--source", action="append", default=[])
    question.add_argument("--date")
    question.set_defaults(func=cmd_question)

    partial = sub.add_parser("partial")
    partial.add_argument("--part", required=True, choices=("concept", "scenario"))
    partial.set_defaults(func=cmd_partial)

    grade = sub.add_parser("grade")
    grade.add_argument("--concept-score", required=True, type=float)
    grade.add_argument("--scenario-score", required=True, type=float)
    grade.add_argument("--explanation-score", required=True, type=float)
    grade.add_argument("--weak-concept", action="append", default=[])
    grade.add_argument("--date")
    grade.set_defaults(func=cmd_grade)

    due = sub.add_parser("due")
    due.add_argument("--date")
    due.set_defaults(func=cmd_due)

    ask_review = sub.add_parser("ask-review")
    ask_review.add_argument("--concept", required=True)
    ask_review.add_argument("--question", required=True)
    ask_review.add_argument("--date")
    ask_review.set_defaults(func=cmd_ask_review)

    review = sub.add_parser("review")
    review.add_argument("--result", required=True, choices=("pass", "fuzzy", "miss"))
    review.add_argument("--date")
    review.set_defaults(func=cmd_review)
    return result


def main() -> None:
    args = parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
