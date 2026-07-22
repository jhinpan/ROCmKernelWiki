#!/usr/bin/env python3
"""Run fixed kernel A/B tasks through external agent and sandbox adapters.

The agent adapter proposes full-file changes. The runner adapter is responsible
for executing trusted task commands inside the MI355 sandbox and returning a
machine-readable correctness/metric/evidence record. Neither adapter receives
GitHub credentials.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
import query as query_tool  # noqa: E402
from _wiki_root import WIKI_ROOT, configure_utf8_stdio  # noqa: E402
from evaluate_kernel_ab import (  # noqa: E402
    evaluate_kernel_results,
    load_kernel_tasks,
)


def _safe_environment() -> dict[str, str]:
    return {
        key: value
        for key, value in os.environ.items()
        if key not in {"GH_TOKEN", "GITHUB_TOKEN", "SSH_AUTH_SOCK"}
    }


def _adapter(
    command: str,
    package: dict[str, Any],
    *,
    cwd: Path,
    timeout: int,
) -> dict[str, Any]:
    result = subprocess.run(
        shlex.split(command),
        cwd=cwd,
        env=_safe_environment(),
        input=json.dumps(package),
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"adapter failed ({result.returncode}): {result.stderr[:500]}"
        )
    payload = json.loads(result.stdout)
    if not isinstance(payload, dict):
        raise ValueError("adapter output must be a JSON object")
    return payload


def _skill_context(task: dict[str, Any]) -> list[dict[str, Any]]:
    pages = query_tool.load_all_pages(use_cache=True)
    idf = query_tool.search_auxiliary_index(pages)["idf"]
    keywords = re.findall(r"[A-Za-z0-9_.+-]+", str(task["prompt"]))
    args = SimpleNamespace(
        type=None,
        tag=None,
        repo=None,
        language=None,
        architecture=task["architecture"],
        symptom=None,
        confidence=None,
        synthesis=True,
        include_out_of_scope=False,
    )
    candidates = query_tool.filter_pages(pages, args)
    for page in candidates:
        page["_score"] = query_tool.score_keyword_match(
            page["fm"],
            page["body"],
            keywords,
            idf=idf,
            ptype=page.get("_ptype", "unknown"),
        )
    candidates = [page for page in candidates if page["_score"] > 0]
    candidates.sort(key=lambda page: (-page["_score"], page["path"]))
    return [
        {
            "id": page["fm"].get("id"),
            "confidence": page["fm"].get("confidence"),
            "sources": page["fm"].get("sources") or [],
            "body": page["body"][:5000],
        }
        for page in candidates[:5]
    ]


def _apply_agent_changes(
    workspace: Path, task: dict[str, Any], payload: dict[str, Any]
) -> None:
    changes = payload.get("changes")
    if not isinstance(changes, list):
        raise ValueError("kernel agent must return a changes list")
    allowed = {Path(path).as_posix() for path in task["allowed_paths"]}
    changed = set()
    for change in changes:
        path = Path(str(change.get("path") or "")).as_posix()
        if path not in allowed:
            raise ValueError(f"kernel agent changed non-allowlisted path {path}")
        content = change.get("content")
        if not isinstance(content, str):
            raise ValueError(f"{path}: content must be text")
        (workspace / path).write_text(content, encoding="utf-8")
        changed.add(path)
    if not changed:
        raise ValueError("kernel agent returned no changes")


def _git(command: list[str], cwd: Path) -> str:
    result = subprocess.run(
        ["git", *command],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr)
    return result.stdout.strip()


def run_ab(args: argparse.Namespace) -> dict[str, Any]:
    tasks = load_kernel_tasks(
        WIKI_ROOT / "data" / "evals" / "kernel-tasks.yaml"
    )
    if args.task not in tasks:
        raise ValueError(f"unknown task {args.task}")
    task = tasks[args.task]
    baseline_sha = args.baseline_sha or _git(["rev-parse", "HEAD"], WIKI_ROOT)
    if not re.fullmatch(r"[0-9a-f]{40}", baseline_sha):
        raise ValueError("baseline SHA must be a full commit hash")
    context = _skill_context(task)
    arms = {}
    with tempfile.TemporaryDirectory(prefix="rocm-wiki-kernel-ab-") as directory:
        root = Path(directory)
        for arm in ("without_skill", "with_skill"):
            workspace = root / arm
            _git(["worktree", "add", "--detach", str(workspace), baseline_sha], WIKI_ROOT)
            try:
                agent_payload = _adapter(
                    args.agent_command,
                    {
                        "mode": arm,
                        "task": task,
                        "baseline_sha": baseline_sha,
                        "context": context if arm == "with_skill" else [],
                        "rules": [
                            "Modify only allowed_paths.",
                            "Preserve correctness before optimizing performance.",
                            "Return full file content, not shell commands.",
                        ],
                    },
                    cwd=workspace,
                    timeout=args.agent_timeout,
                )
                if _git(["status", "--porcelain=v1"], workspace):
                    raise ValueError(
                        "agent adapter mutated the worktree directly; "
                        "it must return bounded full-file changes as JSON"
                    )
                _apply_agent_changes(workspace, task, agent_payload)
                arms[arm] = _adapter(
                    args.runner_command,
                    {
                        "mode": arm,
                        "task": task,
                        "workspace": str(workspace),
                        "baseline_sha": baseline_sha,
                    },
                    cwd=WIKI_ROOT,
                    timeout=args.runner_timeout,
                )
            finally:
                _git(["worktree", "remove", "--force", str(workspace)], WIKI_ROOT)
    fingerprints = {
        str(record.get("environment_fingerprint") or "") for record in arms.values()
    }
    if len(fingerprints) != 1 or not next(iter(fingerprints), ""):
        raise ValueError("both A/B arms must share one environment fingerprint")
    raw = {
        "schema_version": 1,
        "results": [
            {
                "id": args.task,
                "environment_fingerprint": next(iter(fingerprints)),
                "without_skill": arms["without_skill"],
                "with_skill": arms["with_skill"],
            }
        ],
    }
    return {
        "schema_version": 1,
        "baseline_sha": baseline_sha,
        "raw": raw,
        "evaluation": evaluate_kernel_results(tasks, raw),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--baseline-sha")
    parser.add_argument("--agent-command", required=True)
    parser.add_argument("--runner-command", required=True)
    parser.add_argument("--agent-timeout", type=int, default=1800)
    parser.add_argument("--runner-timeout", type=int, default=1800)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        result = run_ab(args)
        args.output.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except (
        OSError,
        RuntimeError,
        ValueError,
        json.JSONDecodeError,
        subprocess.TimeoutExpired,
    ) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    configure_utf8_stdio()
    raise SystemExit(main())
