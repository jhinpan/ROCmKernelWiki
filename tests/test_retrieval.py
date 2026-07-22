#!/usr/bin/env python3
"""Guide-derived retrieval and architecture-safety regression."""

import os
import re
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
CLAIMS = yaml.safe_load(
    (ROOT / "data/guide-claims.yaml").read_text(encoding="utf-8")
)


def _query(claim):
    command = [
        sys.executable,
        "scripts/query.py",
        claim["question"],
        "--limit",
        "5",
        "--compact",
    ]
    if claim.get("synthesis", True):
        command.append("--synthesis")
    environment = os.environ.copy()
    environment["PYTHONUTF8"] = "1"
    result = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=environment,
    )
    assert result.returncode == 0, result.stderr
    return re.findall(r"^  \[[^\]]+\] ([^:]+):", result.stdout, re.MULTILINE)


def _frontmatter_by_id():
    pages = {}
    for base in (ROOT / "wiki", ROOT / "sources"):
        for path in base.rglob("*.md"):
            text = path.read_text(encoding="utf-8")
            if not text.startswith("---\n"):
                continue
            fm = yaml.safe_load(text.split("---", 2)[1])
            if fm and fm.get("id"):
                pages[fm["id"]] = fm
    return pages


def test_guide_retrieval():
    pages = _frontmatter_by_id()
    top1 = 0
    top5 = 0
    safety_violations = []
    misses = []
    for claim in CLAIMS["claims"]:
        hits = _query(claim)
        canonical = claim["canonical_pages"]
        if hits and hits[0] == canonical[0]:
            top1 += 1
        if any(page_id in hits for page_id in canonical):
            top5 += 1
        else:
            misses.append((claim["id"], canonical, hits))

        architecture = claim["architecture"]
        if architecture in {"gfx942", "gfx950"}:
            for page_id in hits:
                archs = set((pages.get(page_id) or {}).get("architectures") or [])
                if archs and architecture not in archs:
                    safety_violations.append((claim["id"], architecture, page_id, archs))

    total = len(CLAIMS["claims"])
    assert total == 43
    assert top1 >= 39, f"top1={top1}/{total}"
    assert top5 == total, f"top5={top5}/{total}; misses={misses}"
    assert not safety_violations, safety_violations


def test_guide_registry_pins_source_and_pages():
    pages = _frontmatter_by_id()
    claims = CLAIMS["claims"]
    source = pages[CLAIMS["source_id"]]
    assert f"/blob/{CLAIMS['pinned_commit']}/" in source["url"]
    assert len({claim["id"] for claim in claims}) == len(claims) == 43
    for claim in claims:
        assert claim["canonical_pages"]
        for page_id in claim["canonical_pages"]:
            assert page_id in pages, (claim["id"], page_id)


def test_unseen_paraphrase_retrieval():
    fixture = yaml.safe_load(
        (ROOT / "data/retrieval-paraphrases.yaml").read_text(encoding="utf-8")
    )
    top1 = 0
    top5 = 0
    misses = []
    for case in fixture["cases"]:
        hits = _query({**case, "architecture": "both"})
        canonical = case["canonical_pages"]
        top1 += bool(hits and hits[0] in canonical)
        matched = any(page_id in hits for page_id in canonical)
        top5 += matched
        if not matched:
            misses.append((case["id"], canonical, hits))
    total = len(fixture["cases"])
    assert total == 15
    assert top1 >= 12, f"paraphrase top1={top1}/{total}"
    assert top5 == total, f"paraphrase top5={top5}/{total}; misses={misses}"


if __name__ == "__main__":
    test_guide_retrieval()
    test_guide_registry_pins_source_and_pages()
    test_unseen_paraphrase_retrieval()
    print("PASS test_guide_retrieval")
    print("PASS test_guide_registry_pins_source_and_pages")
    print("PASS test_unseen_paraphrase_retrieval")
