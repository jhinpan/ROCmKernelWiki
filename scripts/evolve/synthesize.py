#!/usr/bin/env python3
"""Run and validate a bounded synthesis adapter.

The adapter receives JSON on stdin and must return JSON with `summary` and
`changes`. It runs without GitHub credentials. Its output is path-bounded,
machine-authored, and forbidden from self-promoting knowledge to `verified`.
Nothing is written unless --apply is supplied.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from _wiki_root import WIKI_ROOT, configure_utf8_stdio  # noqa: E402
from evolve.schema import load_evolution_schemas  # noqa: E402


def _split_markdown(content: str) -> tuple[dict[str, Any], str]:
    match = re.match(r"^---\s*\r?\n(.*?)\r?\n---\s*\r?\n(.*)", content, re.DOTALL)
    if not match:
        raise ValueError("machine-authored Markdown must have YAML frontmatter")
    frontmatter = yaml.safe_load(match.group(1)) or {}
    if not isinstance(frontmatter, dict):
        raise ValueError("Markdown frontmatter must be an object")
    return frontmatter, match.group(2)


def _render_markdown(frontmatter: dict[str, Any], body: str) -> str:
    return (
        "---\n"
        + yaml.safe_dump(
            frontmatter,
            sort_keys=False,
            allow_unicode=True,
            default_flow_style=False,
        )
        + "---\n"
        + body.lstrip("\n")
    )


def _safe_relative_path(raw_path: str) -> PurePosixPath:
    path = PurePosixPath(raw_path)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ValueError(f"unsafe machine change path: {raw_path}")
    if "\\" in raw_path:
        raise ValueError(f"machine change path must use POSIX separators: {raw_path}")
    return path


def _path_allowed(path: PurePosixPath, schemas: dict[str, Any]) -> bool:
    allowed = (schemas.get("machine_change") or {}).get("allowed_roots") or []
    path_string = path.as_posix()
    for entry in allowed:
        if entry.endswith("/") and path_string.startswith(entry):
            return True
        if path_string == entry:
            return True
    return False


def _prior_frontmatter(root: Path, path: PurePosixPath) -> dict[str, Any]:
    existing = root / path
    if not existing.is_file() or existing.suffix != ".md":
        return {}
    try:
        frontmatter, _ = _split_markdown(existing.read_text(encoding="utf-8"))
        return frontmatter
    except (OSError, ValueError, yaml.YAMLError):
        return {}


def prepare_machine_changes(
    payload: dict[str, Any],
    *,
    root: Path,
    generated_at: str,
) -> list[dict[str, str]]:
    schemas = load_evolution_schemas(root / "data" / "evolution-schemas.yaml")
    if not isinstance(payload, dict) or not isinstance(payload.get("changes"), list):
        raise ValueError("synthesis payload must contain a changes list")
    if not str(payload.get("summary") or "").strip():
        raise ValueError("synthesis payload must contain a summary")
    if len(payload["changes"]) > 20:
        raise ValueError("synthesis output exceeds the 20-file review budget")

    forbidden_confidence = set(
        (schemas.get("machine_change") or {}).get("forbidden_confidence") or []
    )
    prepared = []
    seen_paths = set()
    for index, change in enumerate(payload["changes"]):
        if not isinstance(change, dict):
            raise ValueError(f"changes[{index}] must be an object")
        path = _safe_relative_path(str(change.get("path") or ""))
        if path in seen_paths:
            raise ValueError(f"duplicate machine change path: {path}")
        seen_paths.add(path)
        if not _path_allowed(path, schemas):
            raise ValueError(f"machine change path is not allowlisted: {path}")
        content = change.get("content")
        if not isinstance(content, str):
            raise ValueError(f"{path}: content must be text")

        if path.suffix == ".md":
            frontmatter, body = _split_markdown(content)
            confidence = str(frontmatter.get("confidence") or "")
            if confidence in forbidden_confidence:
                raise ValueError(
                    f"{path}: machine-authored content cannot set confidence={confidence}"
                )
            frontmatter["authored_by"] = "machine"
            candidate_ids = list(payload.get("candidate_ids") or [])
            if candidate_ids:
                frontmatter["generated_from"] = candidate_ids
            if path.parts[0] == "wiki":
                if not confidence:
                    raise ValueError(f"{path}: wiki content must declare confidence")
                prior = _prior_frontmatter(root, path)
                history = list(
                    frontmatter.get("confidence_history")
                    or prior.get("confidence_history")
                    or []
                )
                transition = {
                    "on": generated_at,
                    "from": prior.get("confidence"),
                    "to": confidence,
                    "by": "machine-proposal",
                }
                if not history or history[-1] != transition:
                    history.append(transition)
                frontmatter["confidence_history"] = history
            content = _render_markdown(frontmatter, body)
        elif re.search(r"(?m)^\s*confidence\s*:\s*verified\s*$", content):
            raise ValueError(f"{path}: machine change cannot introduce verified")
        prepared.append({"path": path.as_posix(), "content": content})
    return prepared


def run_agent_command(
    command: str,
    package: dict[str, Any],
    *,
    root: Path,
    timeout_seconds: int,
) -> dict[str, Any]:
    argv = shlex.split(command)
    if not argv:
        raise ValueError("agent command is empty")
    environment = {
        key: value
        for key, value in os.environ.items()
        if key
        not in {
            "GH_TOKEN",
            "GITHUB_TOKEN",
            "SSH_AUTH_SOCK",
            "AWS_ACCESS_KEY_ID",
            "AWS_SECRET_ACCESS_KEY",
        }
    }
    environment["ROCM_WIKI_SYNTHESIS_MODE"] = "untrusted-input"
    status_before = subprocess.run(
        ["git", "status", "--porcelain=v1"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    ).stdout
    result = subprocess.run(
        argv,
        cwd=root,
        env=environment,
        input=json.dumps(package),
        capture_output=True,
        text=True,
        timeout=timeout_seconds,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"synthesis adapter failed ({result.returncode}): {result.stderr[:500]}"
        )
    status_after = subprocess.run(
        ["git", "status", "--porcelain=v1"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    ).stdout
    if status_after != status_before:
        raise RuntimeError(
            "synthesis adapter mutated the checkout directly; return JSON changes only"
        )
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise ValueError("synthesis adapter did not return JSON") from error
    if not isinstance(payload, dict):
        raise ValueError("synthesis adapter output must be an object")
    return payload


def apply_changes(root: Path, changes: list[dict[str, str]]) -> None:
    for change in changes:
        path = root / change["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(change["content"], encoding="utf-8")


def _load_document(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".json":
        document = json.loads(text)
    else:
        document = yaml.safe_load(text)
    if not isinstance(document, dict):
        raise ValueError(f"{path}: expected an object")
    return document


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", type=Path, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--agent-command")
    group.add_argument("--response", type=Path)
    parser.add_argument("--generated-at", required=True)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        package = _load_document(args.package)
        payload = (
            run_agent_command(
                args.agent_command,
                package,
                root=WIKI_ROOT,
                timeout_seconds=args.timeout,
            )
            if args.agent_command
            else _load_document(args.response)
        )
        changes = prepare_machine_changes(
            payload,
            root=WIKI_ROOT,
            generated_at=args.generated_at,
        )
        if args.apply:
            apply_changes(WIKI_ROOT, changes)
        result = {
            "summary": payload["summary"],
            "candidate_ids": payload.get("candidate_ids") or [],
            "changes": changes,
            "applied": bool(args.apply),
        }
        rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
        if args.output:
            args.output.write_text(rendered, encoding="utf-8")
        else:
            print(rendered, end="")
    except (OSError, ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    configure_utf8_stdio()
    raise SystemExit(main())
