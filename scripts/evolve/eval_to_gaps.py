#!/usr/bin/env python3
"""Convert eval misses into a human-reviewed gap queue."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any

import yaml

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from _index import id_index  # noqa: E402
from _wiki_root import WIKI_ROOT, configure_utf8_stdio  # noqa: E402


def retrieval_gaps(result: dict[str, Any], generated_at: str) -> list[dict[str, Any]]:
    known_ids = set(id_index(use_cache=False))
    gaps = []
    for case in result.get("results") or []:
        case_id = str(case.get("id") or "")
        if case.get("expect_refusal"):
            if case.get("refused"):
                continue
            category = "scope-gap"
            reason = "out-of-scope query was not refused"
        elif case.get("recall_at_5") and case.get("source_followed", True):
            continue
        elif not case.get("source_followed", True):
            category = "evidence-gap"
            reason = (
                f"top results did not follow required source "
                f"{case.get('required_source_id')}"
            )
        else:
            canonical = set(case.get("canonical_pages") or [])
            if canonical and canonical <= known_ids:
                category = "ranking-gap"
                reason = "canonical pages exist but did not rank in the top five"
            else:
                category = "page-gap"
                reason = "no resolvable canonical synthesis page exists"
        gaps.append(
            {
                "id": f"eval-{case_id}",
                "generated_at": generated_at,
                "status": "proposed",
                "category": category,
                "eval_case": case_id,
                "reason": reason,
                "canonical_pages": case.get("canonical_pages") or [],
                "observed_top_ids": case.get("top_ids") or [],
            }
        )
    return gaps


def answer_gaps(result: dict[str, Any], generated_at: str) -> list[dict[str, Any]]:
    gaps = []
    for case in result.get("results") or []:
        if case.get("pass"):
            continue
        category = (
            "citation-gap"
            if not case.get("citation_pass")
            else "answer-fact-gap"
        )
        gaps.append(
            {
                "id": f"answer-{case.get('id')}",
                "generated_at": generated_at,
                "status": "proposed",
                "category": category,
                "eval_case": case.get("id"),
                "reason": "answer-level regression",
                "details": {
                    "missing_patterns": case.get("missing_patterns") or [],
                    "forbidden_hits": case.get("forbidden_hits") or [],
                    "missing_citations": case.get("missing_citations") or [],
                    "unresolved_citations": case.get("unresolved_citations") or [],
                },
            }
        )
    return gaps


def write_gap_queue(root: Path, gaps: list[dict[str, Any]]) -> Path:
    path = root / "candidates" / "eval-gaps.yaml"
    existing_document = (
        yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if path.is_file()
        else {}
    )
    existing = {
        gap["id"]: gap
        for gap in existing_document.get("gaps") or []
        if isinstance(gap, dict) and gap.get("id")
    }
    for gap in gaps:
        prior = existing.get(gap["id"])
        if prior and prior.get("status") in {"accepted", "rejected", "superseded"}:
            gap["status"] = prior["status"]
        existing[gap["id"]] = gap
    document = {
        "schema_version": 1,
        "gaps": [existing[key] for key in sorted(existing)],
    }
    path.write_text(
        yaml.safe_dump(document, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrieval", type=Path)
    parser.add_argument("--answers", type=Path)
    parser.add_argument("--generated-at", default=date.today().isoformat())
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not args.retrieval and not args.answers:
        parser.error("provide --retrieval and/or --answers")
    try:
        gaps = []
        if args.retrieval:
            gaps.extend(
                retrieval_gaps(
                    json.loads(args.retrieval.read_text(encoding="utf-8")),
                    args.generated_at,
                )
            )
        if args.answers:
            gaps.extend(
                answer_gaps(
                    json.loads(args.answers.read_text(encoding="utf-8")),
                    args.generated_at,
                )
            )
        if args.dry_run:
            print(yaml.safe_dump({"gaps": gaps}, sort_keys=False), end="")
        else:
            print(write_gap_queue(WIKI_ROOT, gaps))
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    configure_utf8_stdio()
    raise SystemExit(main())
