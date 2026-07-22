#!/usr/bin/env python3
"""Score MI355 kernel-task A/B result bundles without making new claims."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _wiki_root import WIKI_ROOT, configure_utf8_stdio  # noqa: E402


def load_kernel_tasks(path: Path) -> dict[str, dict[str, Any]]:
    document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if document.get("schema_version") != 1:
        raise ValueError(f"{path}: schema_version must be 1")
    tasks = document.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError(f"{path}: tasks must be a non-empty list")
    by_id = {}
    for task in tasks:
        task_id = str(task.get("id") or "")
        if not task_id or task_id in by_id:
            raise ValueError(f"{path}: duplicate/empty task id {task_id!r}")
        if task.get("architecture") != "gfx950":
            raise ValueError(f"{path}: {task_id} must target gfx950")
        if (task.get("metric") or {}).get("direction") not in {"higher", "lower"}:
            raise ValueError(f"{path}: {task_id} metric direction is invalid")
        by_id[task_id] = task
    return by_id


def evaluate_kernel_results(
    tasks: dict[str, dict[str, Any]], document: dict[str, Any]
) -> dict[str, Any]:
    if document.get("schema_version") != 1:
        raise ValueError("kernel result schema_version must be 1")
    results = []
    both_correct = 0
    improved = 0
    for record in document.get("results") or []:
        task_id = str(record.get("id") or "")
        if task_id not in tasks:
            raise ValueError(f"unknown kernel task {task_id!r}")
        without = record.get("without_skill") or {}
        with_skill = record.get("with_skill") or {}
        environment = str(record.get("environment_fingerprint") or "")
        if not environment:
            raise ValueError(f"{task_id}: missing environment_fingerprint")
        correct = bool(without.get("correct")) and bool(with_skill.get("correct"))
        both_correct += correct
        direction = tasks[task_id]["metric"]["direction"]
        baseline = without.get("metric")
        candidate = with_skill.get("metric")
        ratio = None
        did_improve = False
        if correct and isinstance(baseline, (int, float)) and isinstance(
            candidate, (int, float)
        ):
            if baseline == 0 or candidate == 0:
                raise ValueError(f"{task_id}: zero metric cannot form a ratio")
            ratio = (
                candidate / baseline
                if direction == "higher"
                else baseline / candidate
            )
            did_improve = ratio > 1.0
            improved += did_improve
        results.append(
            {
                "id": task_id,
                "both_correct": correct,
                "improvement_ratio": ratio,
                "improved": did_improve,
                "environment_fingerprint": environment,
                "evidence_bundles": {
                    "without_skill": without.get("evidence_bundle"),
                    "with_skill": with_skill.get("evidence_bundle"),
                },
            }
        )
    total = len(results)
    return {
        "schema_version": 1,
        "fixture": bool(document.get("fixture")),
        "metrics": {
            "tasks": total,
            "both_correct_rate": both_correct / total if total else 0.0,
            "improvement_rate_after_correctness": (
                improved / both_correct if both_correct else 0.0
            ),
        },
        "results": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        tasks = load_kernel_tasks(
            WIKI_ROOT / "data" / "evals" / "kernel-tasks.yaml"
        )
        document = json.loads(args.results.read_text(encoding="utf-8"))
        result = evaluate_kernel_results(tasks, document)
        rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
        if args.output:
            args.output.write_text(rendered, encoding="utf-8")
        else:
            print(rendered, end="")
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    configure_utf8_stdio()
    raise SystemExit(main())
