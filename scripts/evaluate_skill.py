#!/usr/bin/env python3
"""Run scored, multi-source retrieval and architecture-safety evaluations."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import query as query_tool  # noqa: E402
from _wiki_root import WIKI_ROOT, configure_utf8_stdio  # noqa: E402


def load_gold_cases(root: Path) -> dict[str, Any]:
    registry = yaml.safe_load(
        (root / "data" / "evals" / "retrieval-gold.yaml").read_text(
            encoding="utf-8"
        )
    )
    if registry.get("schema_version") != 1:
        raise ValueError("retrieval-gold.yaml schema_version must be 1")
    guide = yaml.safe_load(
        (root / "data" / "guide-claims.yaml").read_text(encoding="utf-8")
    )
    paraphrases = yaml.safe_load(
        (root / "data" / "retrieval-paraphrases.yaml").read_text(encoding="utf-8")
    )
    cases = []
    for case in guide.get("claims") or []:
        cases.append({**case, "suite": "guide"})
    for case in paraphrases.get("cases") or []:
        cases.append(
            {
                **case,
                "architecture": case.get("architecture", "both"),
                "suite": "paraphrase",
            }
        )
    for case in registry.get("cases") or []:
        cases.append({**case, "suite": "held-out"})
    ids = [str(case["id"]) for case in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("retrieval gold case IDs must be unique")
    return {
        "schema_version": 1,
        "thresholds": registry.get("thresholds") or {},
        "cases": cases,
    }


def _args(architecture: str | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        type=None,
        tag=None,
        repo=None,
        language=None,
        architecture=architecture,
        symptom=None,
        confidence=None,
        synthesis=False,
        include_out_of_scope=False,
    )


def _dcg(relevance: list[int]) -> float:
    return sum(
        value / math.log2(index + 2)
        for index, value in enumerate(relevance)
    )


def _candidate_pages(
    pages: list[dict[str, Any]],
    inverted: dict[str, set[int]],
    keywords: list[str],
    include_prs: bool,
) -> list[dict[str, Any]]:
    indices = set()
    if include_prs:
        indices.update(
            index
            for keyword in keywords
            for variant in query_tool.expand_keyword(keyword)
            for index in inverted.get(variant.lower(), set())
        )
    # Curated/source-anchor pages are small and always considered. This keeps
    # substring/phrase behavior stable while avoiding scans of unrelated PRs.
    indices.update(
        index
        for index, page in enumerate(pages)
        if query_tool.detect_page_type(page["fm"], page["path"]) != "source-pr"
    )
    return [pages[index] for index in sorted(indices)] if indices else pages


def evaluate_retrieval(root: Path, gold: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    all_pages = query_tool.load_all_pages(use_cache=True)
    all_ids = {
        str(page["fm"].get("id"))
        for page in all_pages
        if page["fm"].get("id")
    }
    auxiliary = query_tool.search_auxiliary_index(all_pages)
    idf = auxiliary["idf"]
    inverted = auxiliary["inverted"]
    results = []
    recall_hits = 0
    reciprocal_rank = 0.0
    ndcg = 0.0
    normal_cases = 0
    refusal_cases = 0
    refusal_hits = 0
    leakage = 0
    architecture_results = 0
    citation_checks = 0
    citation_passes = 0

    for case in gold["cases"]:
        case_started = time.perf_counter()
        architecture = str(case.get("architecture") or "both")
        expect_refusal = bool(case.get("expect_refusal"))
        if expect_refusal:
            refusal_cases += 1
            refused = architecture not in query_tool.in_scope_architectures()
            refusal_hits += refused
            results.append(
                {
                    "id": case["id"],
                    "suite": case.get("suite"),
                    "expect_refusal": True,
                    "refused": refused,
                    "canonical_pages": case.get("canonical_pages") or [],
                    "latency_ms": round(
                        (time.perf_counter() - case_started) * 1000, 3
                    ),
                }
            )
            continue

        normal_cases += 1
        keywords = re.findall(
            r"[A-Za-z0-9_.+-]+", str(case.get("question") or "")
        )
        target_architecture = (
            architecture
            if architecture in query_tool.in_scope_architectures()
            else None
        )
        candidates = _candidate_pages(
            all_pages,
            inverted,
            keywords,
            include_prs=bool(case.get("include_prs")),
        )
        pages = query_tool.filter_pages(candidates, _args(target_architecture))
        for page in pages:
            page["_score"] = query_tool.score_keyword_match(
                page["fm"],
                page["body"],
                keywords,
                idf=idf,
                ptype=page.get("_ptype", "unknown"),
            )
            if target_architecture and target_architecture in (
                page["fm"].get("architectures") or []
            ):
                page["_score"] *= 1.35
        pages = [page for page in pages if page["_score"] > 0]
        pages.sort(key=lambda page: (-page["_score"], page["path"]))
        top = pages[:10]
        top_ids = [str(page["fm"].get("id")) for page in top]
        canonical = [str(value) for value in case.get("canonical_pages") or []]
        ranks = [
            top_ids.index(page_id) + 1
            for page_id in canonical
            if page_id in top_ids
        ]
        rank = min(ranks) if ranks else None
        hit = bool(rank and rank <= 5)
        recall_hits += hit
        reciprocal_rank += 1.0 / rank if rank else 0.0
        relevance = [1 if page_id in canonical else 0 for page_id in top_ids]
        ideal = [1] * min(len(canonical), 10)
        ndcg += (_dcg(relevance) / _dcg(ideal)) if ideal else 1.0

        architecture_violations = []
        if target_architecture:
            for page in top[:5]:
                architectures = set(page["fm"].get("architectures") or [])
                if architectures:
                    architecture_results += 1
                    if target_architecture not in architectures:
                        leakage += 1
                        architecture_violations.append(page["fm"].get("id"))

        unresolved = []
        for page in top[:5]:
            for source_id in page["fm"].get("sources") or []:
                citation_checks += 1
                if source_id in all_ids:
                    citation_passes += 1
                else:
                    unresolved.append(source_id)
        required_source = case.get("source_id")
        source_followed = True
        if required_source:
            source_followed = any(
                page["fm"].get("id") == required_source
                or required_source in (page["fm"].get("sources") or [])
                for page in top[:5]
            )
        results.append(
            {
                "id": case["id"],
                "suite": case.get("suite"),
                "rank": rank,
                "recall_at_5": hit,
                "top_ids": top_ids,
                "canonical_pages": canonical,
                "architecture_violations": architecture_violations,
                "unresolved_citations": sorted(set(unresolved)),
                "source_followed": source_followed,
                "required_source_id": required_source,
                "latency_ms": round(
                    (time.perf_counter() - case_started) * 1000, 3
                ),
            }
        )

    metrics = {
        "cases": len(gold["cases"]),
        "retrieval_cases": normal_cases,
        "refusal_cases": refusal_cases,
        "recall_at_5": recall_hits / normal_cases if normal_cases else 1.0,
        "mrr_at_10": reciprocal_rank / normal_cases if normal_cases else 1.0,
        "ndcg_at_10": ndcg / normal_cases if normal_cases else 1.0,
        "refusal_accuracy": refusal_hits / refusal_cases if refusal_cases else 1.0,
        "architecture_leakage_rate": (
            leakage / architecture_results if architecture_results else 0.0
        ),
        "citation_resolvability_rate": (
            citation_passes / citation_checks if citation_checks else 1.0
        ),
        "index_terms": len(inverted),
        "duration_seconds": round(time.perf_counter() - started, 3),
    }
    return {"schema_version": 1, "metrics": metrics, "results": results}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    try:
        gold = load_gold_cases(WIKI_ROOT)
        result = evaluate_retrieval(WIKI_ROOT, gold)
        rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
        if args.output:
            args.output.write_text(rendered, encoding="utf-8")
        else:
            print(rendered, end="")
        if args.check:
            for metric, threshold in gold["thresholds"].items():
                observed = float(result["metrics"].get(metric, 0))
                if metric == "architecture_leakage_rate":
                    passed = observed <= float(threshold)
                    relation = "<="
                else:
                    passed = observed >= float(threshold)
                    relation = ">="
                if not passed:
                    print(
                        f"ERROR: {metric}={observed} must be {relation} {threshold}",
                        file=sys.stderr,
                    )
                    return 1
    except (OSError, ValueError, yaml.YAMLError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    configure_utf8_stdio()
    raise SystemExit(main())
