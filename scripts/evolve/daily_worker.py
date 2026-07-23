#!/usr/bin/env python3
"""Run the daily evidence refresh in a disposable bot clone."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from datetime import date
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from _wiki_root import WIKI_ROOT, configure_utf8_stdio  # noqa: E402
from evolve.draft_pr import rolling_branch  # noqa: E402


def _run(
    command: list[str],
    *,
    cwd: Path,
    environment: dict[str, str] | None = None,
) -> str:
    result = subprocess.run(
        command,
        cwd=cwd,
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


def _remote_branch_exists(remote: str, branch: str, cwd: Path) -> bool:
    output = _run(
        ["git", "ls-remote", "--heads", remote, f"refs/heads/{branch}"],
        cwd=cwd,
    )
    return bool(output)


def _configure_bot_identity(clone: Path) -> None:
    _run(
        [
            "git",
            "config",
            "user.name",
            os.environ.get("WIKI_BOT_NAME", "rocmkernelwiki-evolution[bot]"),
        ],
        cwd=clone,
    )
    _run(
        [
            "git",
            "config",
            "user.email",
            os.environ.get(
                "WIKI_BOT_EMAIL",
                "rocmkernelwiki-evolution[bot]@users.noreply.github.com",
            ),
        ],
        cwd=clone,
    )


def _sync_with_base(
    clone: Path,
    *,
    base: str,
    branch: str,
    source_branch: str,
) -> None:
    if source_branch != branch:
        _run(["git", "switch", "-c", branch], cwd=clone)
        return
    # A rolling branch owns the discovery watermarks, but executable controller
    # code and tests must always come from the latest protected base branch.
    _run(
        [
            "git",
            "fetch",
            "origin",
            f"+{base}:refs/remotes/origin/{base}",
        ],
        cwd=clone,
    )
    try:
        _run(["git", "rebase", f"origin/{base}"], cwd=clone)
    except RuntimeError:
        # The clone is disposable, but abort explicitly so diagnostics see a
        # clean branch and no caller can accidentally continue in a conflicted
        # rebase state.
        subprocess.run(
            ["git", "rebase", "--abort"],
            cwd=clone,
            capture_output=True,
            text=True,
            check=False,
        )
        raise


def run_daily(args: argparse.Namespace) -> dict[str, str]:
    run_date = args.run_date or date.today().isoformat()
    branch = args.branch or rolling_branch(run_date)
    remote = args.remote or _run(
        ["git", "remote", "get-url", "origin"], cwd=WIKI_ROOT
    )
    with tempfile.TemporaryDirectory(prefix="rocm-wiki-evolution-") as directory:
        clone = Path(directory) / "repo"
        source_branch = (
            branch
            if _remote_branch_exists(remote, branch, WIKI_ROOT)
            else args.base
        )
        _run(
            [
                "git",
                "clone",
                "--filter=blob:none",
                "--branch",
                source_branch,
                remote,
                str(clone),
            ],
            cwd=Path(directory),
        )
        _configure_bot_identity(clone)
        _sync_with_base(
            clone,
            base=args.base,
            branch=branch,
            source_branch=source_branch,
        )

        refresh = [
            sys.executable,
            "scripts/evolve/refresh.py",
            "--captured-at",
            run_date,
            "--max-files",
            str(args.max_files),
            "--max-lines",
            str(args.max_lines),
        ]
        if args.since:
            refresh.extend(["--since", args.since])
        if args.bootstrap_trees:
            refresh.append("--bootstrap-trees")
        for source_id in args.source_ids or []:
            refresh.extend(["--source", source_id])
        refresh_result = json.loads(_run(refresh, cwd=clone))
        run_id = str(refresh_result["run_id"])

        if args.agent_command:
            proposals = clone / "candidates" / "synthesis-proposals.yaml"
            if proposals.is_file():
                _run(
                    [
                        sys.executable,
                        "scripts/evolve/synthesize.py",
                        "--package",
                        str(proposals),
                        "--agent-command",
                        args.agent_command,
                        "--generated-at",
                        run_date,
                        "--apply",
                    ],
                    cwd=clone,
                )
                _run(
                    [
                        sys.executable,
                        "scripts/evolve/refresh.py",
                        "--captured-at",
                        run_date,
                        "--run-id",
                        run_id,
                        "--skip-discovery",
                        "--allow-dirty",
                        "--max-files",
                        str(args.max_files),
                        "--max-lines",
                        str(args.max_lines),
                    ],
                    cwd=clone,
                )

        summary = (
            clone
            / "candidates"
            / "runs"
            / run_id
            / "refresh-summary.yaml"
        )
        publish = [
            sys.executable,
            "scripts/evolve/draft_pr.py",
            "--summary",
            str(summary),
            "--base",
            args.base,
            "--branch",
            branch,
        ]
        if args.dry_run:
            publish.append("--dry-run")
        output = _run(publish, cwd=clone)
        result = json.loads(output)
        return {
            "run_date": run_date,
            "run_id": run_id,
            "branch": branch,
            "status": str(result.get("status")),
            "url": str(result.get("url") or ""),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-date")
    parser.add_argument("--remote")
    parser.add_argument("--base", default="main")
    parser.add_argument("--branch")
    parser.add_argument("--source", action="append", dest="source_ids")
    parser.add_argument(
        "--since", default=os.environ.get("ROCM_WIKI_INITIAL_SINCE")
    )
    parser.add_argument(
        "--bootstrap-trees",
        action="store_true",
        default=os.environ.get("ROCM_WIKI_BOOTSTRAP_TREES") == "1",
    )
    parser.add_argument(
        "--agent-command",
        default=os.environ.get("ROCM_WIKI_AGENT_COMMAND"),
    )
    parser.add_argument("--max-files", type=int, default=100)
    parser.add_argument("--max-lines", type=int, default=10000)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    try:
        result = run_daily(parse_args())
    except (OSError, RuntimeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    configure_utf8_stdio()
    raise SystemExit(main())
