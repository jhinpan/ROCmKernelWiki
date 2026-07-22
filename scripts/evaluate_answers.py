#!/usr/bin/env python3
"""Score answer-level facts, architecture safety, and citations offline."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _index import id_index  # noqa: E402
from _wiki_root import WIKI_ROOT, configure_utf8_stdio  # noqa: E402


def load_answer_gold(path: Path) -> dict[str, Any]:
    document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if document.get("schema_version") != 1:
        raise ValueError(f"{path}: schema_version must be 1")
    cases = document.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError(f"{path}: cases must be a non-empty list")
    return document


def evaluate_answer_records(
    gold: dict[str, Any],
    records: list[dict[str, Any]],
    *,
    root: Path,
) -> dict[str, Any]:
    records_by_id = {
        str(record.get("id")): record
        for record in records
        if isinstance(record, dict) and record.get("id")
    }
    resolvable = set(id_index(use_cache=False))
    results = []
    passed = 0
    citation_passed = 0
    for case in gold["cases"]:
        case_id = str(case["id"])
        record = records_by_id.get(case_id) or {}
        answer = str(record.get("answer") or "")
        citations = [
            str(citation) for citation in (record.get("citations") or [])
        ]
        missing_patterns = [
            pattern
            for pattern in case.get("required_patterns") or []
            if not re.search(pattern, answer, re.IGNORECASE | re.DOTALL)
        ]
        forbidden_hits = [
            pattern
            for pattern in case.get("forbidden_patterns") or []
            if re.search(pattern, answer, re.IGNORECASE | re.DOTALL)
        ]
        unresolved = [citation for citation in citations if citation not in resolvable]
        required_citations = set(case.get("required_citations") or [])
        missing_citations = sorted(required_citations - set(citations))
        citation_ok = not unresolved and not missing_citations
        fact_ok = bool(answer) and not missing_patterns and not forbidden_hits
        case_ok = fact_ok and citation_ok
        passed += case_ok
        citation_passed += citation_ok
        results.append(
            {
                "id": case_id,
                "pass": case_ok,
                "fact_pass": fact_ok,
                "citation_pass": citation_ok,
                "missing_patterns": missing_patterns,
                "forbidden_hits": forbidden_hits,
                "missing_citations": missing_citations,
                "unresolved_citations": unresolved,
            }
        )
    total = len(gold["cases"])
    return {
        "schema_version": 1,
        "metrics": {
            "cases": total,
            "case_pass_rate": passed / total,
            "citation_validity_rate": citation_passed / total,
        },
        "results": results,
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gold",
        type=Path,
        default=WIKI_ROOT / "data" / "evals" / "answer-gold.yaml",
    )
    parser.add_argument("--answers", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    try:
        gold = load_answer_gold(args.gold)
        result = evaluate_answer_records(
            gold,
            _read_jsonl(args.answers),
            root=WIKI_ROOT,
        )
        rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
        if args.output:
            args.output.write_text(rendered, encoding="utf-8")
        else:
            print(rendered, end="")
        if args.check:
            for metric, threshold in (gold.get("thresholds") or {}).items():
                if result["metrics"].get(metric, 0) < float(threshold):
                    print(
                        f"ERROR: {metric}={result['metrics'].get(metric)} "
                        f"< {threshold}",
                        file=sys.stderr,
                    )
                    return 1
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    configure_utf8_stdio()
    raise SystemExit(main())
