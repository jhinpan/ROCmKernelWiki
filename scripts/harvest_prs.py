#!/usr/bin/env python3
"""Compatibility entrypoint for incremental PR discovery.

The historical implementation rescanned every merged PR and rewrote source
pages from scratch. That could erase artifact, facet, and PR↔wiki enrichment.
This wrapper delegates to scripts/evolve/discover.py, which is incremental and
merge-safe.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _wiki_root import WIKI_ROOT, configure_utf8_stdio  # noqa: E402
from evolve.discover import run_discovery  # noqa: E402
from evolve.registry import active_sources, load_registry  # noqa: E402

LEGACY_SHORTS = {
    "composable_kernel": "rocm-rocm-libraries",
    "hipBLASLt": "rocm-rocm-libraries",
    "Tensile": "rocm-rocm-libraries",
    "rocBLAS": "rocm-rocm-libraries",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", help="source short name or source id")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--since", help="required on first live run")
    parser.add_argument("--until")
    parser.add_argument("--max", type=int, default=200, dest="max_items")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--captured-at")
    args = parser.parse_args()
    if not args.all and not args.repo:
        parser.error("specify --repo or --all")

    registry = load_registry(WIKI_ROOT / "data" / "sources.yaml")
    pr_sources = [
        source
        for source in active_sources(registry)
        if source["kind"] == "github-prs"
    ]
    source_ids = None
    if args.repo:
        requested = LEGACY_SHORTS.get(args.repo, args.repo)
        matches = [
            source["id"]
            for source in pr_sources
            if requested in {source["id"], source["short"]}
        ]
        if not matches:
            print(f"ERROR: unknown active source '{args.repo}'", file=sys.stderr)
            return 2
        source_ids = matches

    try:
        summary = run_discovery(
            root=WIKI_ROOT,
            source_ids=source_ids,
            captured_at=args.captured_at,
            since=args.since,
            until=args.until,
            max_items=args.max_items,
            dry_run=args.dry_run,
        )
    except (OSError, RuntimeError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    configure_utf8_stdio()
    raise SystemExit(main())
