#!/usr/bin/env python3
"""Commit a bounded refresh and open or update its rolling Draft PR."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import date
from pathlib import Path
from typing import Any

import yaml

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from _wiki_root import WIKI_ROOT, configure_utf8_stdio  # noqa: E402


def rolling_branch(run_date: str) -> str:
    # One rolling branch preserves discovery watermarks while review is open;
    # a fresh branch is created from main after the prior PR is merged.
    date.fromisoformat(run_date)
    return "bot/evolution"


def build_pr_body(summary: dict[str, Any]) -> str:
    discovery = summary.get("discovery") or {}
    files = summary.get("changed_files") or []
    displayed = "\n".join(f"- `{path}`" for path in files[:30])
    if len(files) > 30:
        displayed += f"\n- … and {len(files) - 30} more"
    return f"""## Evidence refresh

- Run date: `{summary.get('run_date')}`
- Included / deferred / excluded / quarantined:
  `{discovery.get('included', 0)} / {discovery.get('deferred', 0)} /
  {discovery.get('excluded', 0)} / {discovery.get('quarantined', 0)}`
- Gap proposals: `{summary.get('gap_proposals', 0)}`
- Machine-authored changes: `{summary.get('machine_changes', 0)}`
- Diff budget: `{len(files)} files`, `{summary.get('changed_lines', 0)} lines`

## Changed files

{displayed or '- No material changes'}

## Review contract

- This PR is intentionally draft and cannot approve or merge itself.
- Upstream PR/blog text is untrusted data, not agent instructions.
- Hardware or performance confidence cannot be promoted without a linked,
  immutable MI355 evidence bundle.
- Generated indices, provenance checks, retrieval evals, and schema validation
  must pass before human review.
"""


def _run(
    command: list[str],
    *,
    root: Path,
    environment: dict[str, str] | None = None,
) -> str:
    result = subprocess.run(
        command,
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"{' '.join(command)} failed:\n{result.stdout}\n{result.stderr}"
        )
    return result.stdout.strip()


def publish_draft(
    *,
    root: Path,
    summary: dict[str, Any],
    base: str,
    branch: str,
    dry_run: bool,
) -> dict[str, Any]:
    current = _run(["git", "branch", "--show-current"], root=root)
    if current != branch:
        raise ValueError(
            f"dedicated bot checkout must be on {branch}, found {current}"
        )
    status = _run(["git", "status", "--porcelain=v1"], root=root)
    if not status:
        return {"status": "no-op", "branch": branch}
    body = build_pr_body(summary)
    title = f"chore(wiki): evidence refresh {summary['run_date']}"
    if dry_run:
        return {
            "status": "dry-run",
            "branch": branch,
            "title": title,
            "body": body,
        }

    _run(["git", "add", "--all"], root=root)
    _run(
        [
            "git",
            "commit",
            "-m",
            f"chore(wiki): refresh evidence {summary['run_date']}",
        ],
        root=root,
    )
    _run(["git", "push", "--force-with-lease", "-u", "origin", branch], root=root)
    existing = _run(
        [
            "gh",
            "pr",
            "list",
            "--state",
            "open",
            "--head",
            branch,
            "--json",
            "number,url",
        ],
        root=root,
    )
    pulls = json.loads(existing or "[]")
    if pulls:
        number = str(pulls[0]["number"])
        _run(
            ["gh", "pr", "edit", number, "--title", title, "--body", body],
            root=root,
        )
        url = pulls[0]["url"]
        action = "updated"
    else:
        url = _run(
            [
                "gh",
                "pr",
                "create",
                "--draft",
                "--base",
                base,
                "--head",
                branch,
                "--title",
                title,
                "--body",
                body,
            ],
            root=root,
        )
        action = "created"
    return {"status": action, "branch": branch, "url": url}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--base", default="main")
    parser.add_argument("--branch")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    try:
        summary = yaml.safe_load(args.summary.read_text(encoding="utf-8")) or {}
        branch = args.branch or rolling_branch(str(summary["run_date"]))
        result = publish_draft(
            root=WIKI_ROOT,
            summary=summary,
            base=args.base,
            branch=branch,
            dry_run=args.dry_run,
        )
    except (OSError, KeyError, RuntimeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    configure_utf8_stdio()
    raise SystemExit(main())
