#!/usr/bin/env python3
"""Detect evidence clusters that lack a covering synthesis page."""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Any

import yaml

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from _wiki_root import WIKI_ROOT, configure_utf8_stdio  # noqa: E402
from evolve.schema import validate_proposal  # noqa: E402


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def _cluster_key(candidate: dict[str, Any]) -> tuple[str, str] | None:
    architectures = candidate.get("architectures") or []
    if len(architectures) != 1:
        return None
    concepts = list(candidate.get("kernel_types") or [])
    concepts.extend(candidate.get("hardware_features") or [])
    if not concepts:
        return None
    return str(architectures[0]), str(concepts[0])


def detect_gap_proposals(
    candidates: list[dict[str, Any]],
    *,
    covered_facets: set[str],
    generated_at: str,
    minimum_cluster: int = 3,
) -> list[dict[str, Any]]:
    clusters: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        if candidate.get("decision") not in {"include", "defer", "quarantine"}:
            continue
        key = _cluster_key(candidate)
        if key is not None:
            clusters[key].append(candidate)

    proposals = []
    for (architecture, facet), members in sorted(clusters.items()):
        coverage_key = f"{architecture}:{facet}"
        if coverage_key in covered_facets or len(members) < minimum_cluster:
            continue
        proposal = {
            "id": f"gap-{_slug(architecture)}-{_slug(facet)}",
            "generated_at": generated_at,
            "status": "proposed",
            "candidate_ids": sorted(
                str(member.get("id")) for member in members if member.get("id")
            ),
            "proposed_action": "create-page",
            "proposed_title": f"{facet} on {architecture}",
            "affected_pages": [],
            "confidence_proposal": "inferred",
            "experiment_request": {
                "architecture": architecture,
                "facet": facet,
                "candidate_count": len(members),
                "hardware_required": any(
                    member.get("scope_status") == "quarantine" for member in members
                ),
            },
            "reason": (
                f"{len(members)} related evidence candidates have no synthesis "
                f"page covering {coverage_key}"
            ),
        }
        proposals.append(proposal)
    return proposals


def load_candidates(root: Path) -> list[dict[str, Any]]:
    candidates = []
    for ledger in sorted((root / "candidates" / "runs").glob("*/*.yaml")):
        if ledger.name == "manifest.yaml":
            continue
        document = yaml.safe_load(ledger.read_text(encoding="utf-8")) or {}
        candidates.extend(
            item
            for item in (document.get("candidates") or [])
            if isinstance(item, dict)
        )
    # Include legacy deferred ledgers so the historical backlog is not lost.
    for ledger in sorted((root / "candidates").glob("*.yaml")):
        document = yaml.safe_load(ledger.read_text(encoding="utf-8")) or {}
        repo = str(document.get("repo") or "")
        short = ledger.stem
        for item in document.get("prs") or []:
            if not isinstance(item, dict) or item.get("decision") != "defer":
                continue
            candidates.append(
                {
                    "id": f"legacy-{short}:pr:{item.get('pr')}",
                    "source_id": f"legacy-{short}",
                    "source_kind": "github-pr",
                    "repo": repo,
                    "decision": "defer",
                    "title": item.get("title"),
                    "architectures": [],
                    "kernel_types": [],
                    "hardware_features": [],
                }
            )
    return candidates


def wiki_coverage(root: Path) -> set[str]:
    coverage = set()
    for path in (root / "wiki").rglob("*.md"):
        text = path.read_text(encoding="utf-8")
        match = re.match(
            r"^---\s*\r?\n(.*?)\r?\n---\s*\r?\n", text, re.DOTALL
        )
        if not match:
            continue
        frontmatter = yaml.safe_load(match.group(1)) or {}
        concepts = list(frontmatter.get("kernel_types") or [])
        concepts.extend(frontmatter.get("hardware_features") or [])
        for architecture in frontmatter.get("architectures") or []:
            for concept in concepts:
                coverage.add(f"{architecture}:{concept}")
    return coverage


def write_proposals(
    root: Path, proposals: list[dict[str, Any]], generated_at: str
) -> Path:
    path = root / "candidates" / "synthesis-proposals.yaml"
    existing_document = (
        yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        if path.exists()
        else {}
    )
    existing = {
        proposal["id"]: proposal
        for proposal in existing_document.get("proposals") or []
        if isinstance(proposal, dict) and proposal.get("id")
    }
    for proposal in proposals:
        prior = existing.get(proposal["id"])
        if prior and prior.get("status") in {"accepted", "rejected", "superseded"}:
            proposal["status"] = prior["status"]
        existing[proposal["id"]] = proposal
    document = {
        "schema_version": 1,
        "generated_at": generated_at,
        "proposals": [existing[key] for key in sorted(existing)],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(document, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--minimum-cluster", type=int, default=3)
    parser.add_argument("--generated-at", default=date.today().isoformat())
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    proposals = detect_gap_proposals(
        load_candidates(WIKI_ROOT),
        covered_facets=wiki_coverage(WIKI_ROOT),
        generated_at=args.generated_at,
        minimum_cluster=args.minimum_cluster,
    )
    schemas = WIKI_ROOT / "data" / "evolution-schemas.yaml"
    errors = [
        f"{proposal['id']}: {error}"
        for proposal in proposals
        for error in validate_proposal(proposal, schemas)
    ]
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    if args.dry_run:
        print(yaml.safe_dump({"proposals": proposals}, sort_keys=False), end="")
    else:
        print(write_proposals(WIKI_ROOT, proposals, args.generated_at))
    return 0


if __name__ == "__main__":
    configure_utf8_stdio()
    raise SystemExit(main())
