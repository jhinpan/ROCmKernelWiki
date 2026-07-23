#!/usr/bin/env python3
"""Run one bounded, reviewable evolution refresh."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from datetime import date
from pathlib import Path
from typing import Any

import yaml

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from _wiki_root import WIKI_ROOT, configure_utf8_stdio  # noqa: E402
from evolve.corpus import write_manifest  # noqa: E402
from evolve.discover import (  # noqa: E402
    new_run_id,
    run_discovery,
    validate_run_id,
)
from evolve.gaps import (  # noqa: E402
    detect_gap_proposals,
    load_candidates,
    wiki_coverage,
    write_proposals,
)
from evolve.synthesize import (  # noqa: E402
    _load_document,
    apply_changes,
    prepare_machine_changes,
)


def enforce_change_budget(
    changed_files: list[str],
    *,
    changed_lines: int,
    max_files: int,
    max_lines: int,
) -> None:
    if len(changed_files) > max_files:
        raise ValueError(
            f"file budget exceeded: {len(changed_files)} > {max_files}"
        )
    if changed_lines > max_lines:
        raise ValueError(
            f"line budget exceeded: {changed_lines} > {max_lines}"
        )


def _run(command: list[str], root: Path) -> str:
    result = subprocess.run(
        command,
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"{' '.join(command)} failed:\n{result.stdout}\n{result.stderr}"
        )
    return result.stdout


def _git_changes(root: Path) -> tuple[list[str], int]:
    output = _run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        root,
    )
    files = []
    for line in output.splitlines():
        if not line:
            continue
        path = line[3:]
        if " -> " in path:
            path = path.split(" -> ", 1)[1]
        files.append(path)
    numstat = _run(["git", "diff", "--numstat"], root)
    changed_lines = 0
    tracked = set()
    for line in numstat.splitlines():
        additions, deletions, path = line.split("\t", 2)
        tracked.add(path)
        if additions.isdigit():
            changed_lines += int(additions)
        if deletions.isdigit():
            changed_lines += int(deletions)
    for path in files:
        candidate = root / path
        if path not in tracked and candidate.is_file():
            try:
                changed_lines += len(candidate.read_text(encoding="utf-8").splitlines())
            except UnicodeDecodeError:
                changed_lines += 1
    return sorted(set(files)), changed_lines


def _write_summary(root: Path, run_id: str, summary: dict[str, Any]) -> Path:
    path = root / "candidates" / "runs" / run_id / "refresh-summary.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(summary, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return path


def _finalize_summary(
    root: Path,
    run_id: str,
    summary: dict[str, Any],
    *,
    max_files: int,
    max_lines: int,
    max_passes: int = 5,
) -> dict[str, Any]:
    """Write and measure until summary metadata matches the on-disk diff.

    The summary includes its own path and line contribution, so one measurement
    cannot be final. A bounded fixed-point loop makes the invariant explicit:
    budgets are enforced only when the values stored in the summary equal a
    fresh measurement of the exact state that contains that summary.
    """
    for _ in range(max_passes):
        _write_summary(root, run_id, summary)
        changed_files, changed_lines = _git_changes(root)
        if (
            summary.get("changed_files") == changed_files
            and summary.get("changed_lines") == changed_lines
        ):
            enforce_change_budget(
                changed_files,
                changed_lines=changed_lines,
                max_files=max_files,
                max_lines=max_lines,
            )
            return summary
        summary["changed_files"] = changed_files
        summary["changed_lines"] = changed_lines
    raise RuntimeError("refresh summary accounting did not converge")


def run_refresh(args: argparse.Namespace, root: Path = WIKI_ROOT) -> dict[str, Any]:
    run_date = args.captured_at or date.today().isoformat()
    run_id = validate_run_id(getattr(args, "run_id", None) or new_run_id())
    initial_files, _ = _git_changes(root)
    if initial_files and not args.allow_dirty:
        raise ValueError(
            "refresh requires a clean checkout; use a dedicated bot clone "
            f"(existing changes: {', '.join(initial_files[:5])})"
        )

    discovery = {
        "included": 0,
        "deferred": 0,
        "excluded": 0,
        "quarantined": 0,
        "total": 0,
        "run_id": run_id,
        "run_date": run_date,
    }
    if not args.skip_discovery:
        discovery = run_discovery(
            root=root,
            source_ids=args.source_ids,
            fixture_path=args.fixture,
            captured_at=run_date,
            run_id=run_id,
            since=args.since,
            until=args.until,
            max_items=args.max_items,
            bootstrap_trees=args.bootstrap_trees,
            dry_run=args.dry_run,
        )

    proposals = detect_gap_proposals(
        load_candidates(root),
        covered_facets=wiki_coverage(root),
        generated_at=run_date,
        minimum_cluster=args.minimum_cluster,
    )
    if not args.dry_run:
        write_proposals(root, proposals, run_date)

    synthesis_count = 0
    if args.synthesis_response:
        payload = _load_document(args.synthesis_response)
        changes = prepare_machine_changes(
            payload,
            root=root,
            generated_at=run_date,
        )
        synthesis_count = len(changes)
        if not args.dry_run:
            apply_changes(root, changes)

    if not args.dry_run:
        _run([sys.executable, "scripts/generate-indices.py"], root)
        write_manifest(root)
        with tempfile.TemporaryDirectory(prefix="rocm-wiki-eval-") as directory:
            retrieval_result = Path(directory) / "retrieval.json"
            _run(
                [
                    sys.executable,
                    "scripts/evaluate_skill.py",
                    "--output",
                    str(retrieval_result),
                    "--check",
                ],
                root,
            )
            _run(
                [
                    sys.executable,
                    "scripts/evolve/eval_to_gaps.py",
                    "--retrieval",
                    str(retrieval_result),
                    "--generated-at",
                    run_date,
                ],
                root,
            )

    changed_files, changed_lines = _git_changes(root)
    enforce_change_budget(
        changed_files,
        changed_lines=changed_lines,
        max_files=args.max_files,
        max_lines=args.max_lines,
    )
    summary = {
        "schema_version": 1,
        "run_id": run_id,
        "run_date": run_date,
        "discovery": discovery,
        "gap_proposals": len(proposals),
        "machine_changes": synthesis_count,
        "changed_files": changed_files,
        "changed_lines": changed_lines,
        "dry_run": bool(args.dry_run),
    }
    if not args.dry_run:
        summary = _finalize_summary(
            root,
            run_id,
            summary,
            max_files=args.max_files,
            max_lines=args.max_lines,
        )
        _run([sys.executable, "scripts/validate.py"], root)
        _run([sys.executable, "tests/test_evolution.py"], root)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", dest="source_ids")
    parser.add_argument("--fixture", type=Path)
    parser.add_argument("--captured-at")
    parser.add_argument("--run-id")
    parser.add_argument("--since")
    parser.add_argument("--until")
    parser.add_argument("--max-items", type=int)
    parser.add_argument("--minimum-cluster", type=int, default=3)
    parser.add_argument("--max-files", type=int, default=100)
    parser.add_argument("--max-lines", type=int, default=10000)
    parser.add_argument("--bootstrap-trees", action="store_true")
    parser.add_argument("--skip-discovery", action="store_true")
    parser.add_argument("--synthesis-response", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        summary = run_refresh(args)
    except (OSError, RuntimeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    configure_utf8_stdio()
    raise SystemExit(main())
