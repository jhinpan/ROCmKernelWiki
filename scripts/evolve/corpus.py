#!/usr/bin/env python3
"""Generate the deterministic corpus inventory manifest."""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path
from typing import Any

import yaml

SCRIPTS_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from _wiki_root import WIKI_ROOT, configure_utf8_stdio  # noqa: E402


def _frontmatter(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8-sig")
    match = re.match(r"^---\s*\r?\n(.*?)\r?\n---\s*\r?\n", text, re.DOTALL)
    if not match:
        return {}
    parsed = yaml.safe_load(match.group(1)) or {}
    return parsed if isinstance(parsed, dict) else {}


def _active_wiki(frontmatter: dict[str, Any], scope: dict[str, Any]) -> bool:
    if frontmatter.get("id") in set(scope.get("quarantined_pages") or []):
        return False
    active_architectures = set(scope.get("in_scope_architectures") or [])
    architectures = set(frontmatter.get("architectures") or [])
    return not architectures or architectures <= active_architectures


def build_manifest(root: Path) -> dict[str, Any]:
    source_prs = sorted((root / "sources" / "prs").glob("*/*.md"))
    wiki_pages = sorted((root / "wiki").rglob("*.md"))
    docs = sorted((root / "sources" / "docs").glob("*.md"))
    blogs = sorted((root / "sources" / "blogs").glob("*.md"))
    refs = sorted((root / "sources" / "refs").glob("*.md"))
    all_pages = source_prs + wiki_pages + docs + blogs + refs
    scope = yaml.safe_load(
        (root / "data" / "scope.yaml").read_text(encoding="utf-8")
    )
    active_wiki_pages = sum(
        _active_wiki(_frontmatter(path), scope) for path in wiki_pages
    )
    ids = {
        frontmatter["id"]
        for path in all_pages
        if (frontmatter := _frontmatter(path)).get("id")
    }
    artifact_bundles = len(
        list((root / "artifacts" / "prs").glob("*/PR-*/PROVENANCE.yaml"))
    )
    examples = sum(
        directory.is_dir()
        and (directory / "README.md").is_file()
        and (directory / "build.sh").is_file()
        for directory in (root / "examples").iterdir()
    )
    cutoff = yaml.safe_load(
        (root / "data" / "refresh-cutoff.yaml").read_text(encoding="utf-8")
    )
    registry_path = root / "data" / "sources.yaml"
    return {
        "schema_version": 1,
        "counts": {
            "source_prs": len(source_prs),
            "wiki_pages": len(wiki_pages),
            "active_wiki_pages": active_wiki_pages,
            "quarantined_wiki_pages": len(wiki_pages) - active_wiki_pages,
            "docs_and_blogs": len(docs) + len(blogs),
            "reference_repositories": len(refs),
            "artifact_bundles": artifact_bundles,
            "examples": examples,
            "unique_page_ids": len(ids),
        },
        "cutoffs": {
            "merged_prs": str(cutoff.get("cutoff_date")),
            "harvested_at": str(cutoff.get("harvested_at")),
        },
        "source_registry_sha256": (
            hashlib.sha256(registry_path.read_bytes()).hexdigest()
            if registry_path.is_file()
            else None
        ),
    }


def write_manifest(root: Path) -> Path:
    destination = root / "data" / "corpus-manifest.yaml"
    destination.write_text(
        yaml.safe_dump(
            build_manifest(root),
            sort_keys=False,
            allow_unicode=True,
            default_flow_style=False,
        ),
        encoding="utf-8",
    )
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    expected = build_manifest(WIKI_ROOT)
    destination = WIKI_ROOT / "data" / "corpus-manifest.yaml"
    if args.check:
        actual = (
            yaml.safe_load(destination.read_text(encoding="utf-8"))
            if destination.is_file()
            else None
        )
        if actual != expected:
            print("ERROR: data/corpus-manifest.yaml is stale", file=sys.stderr)
            return 1
        print("OK: corpus manifest is current")
        return 0
    if args.write:
        print(write_manifest(WIKI_ROOT))
        return 0
    print(
        yaml.safe_dump(
            expected,
            sort_keys=False,
            allow_unicode=True,
            default_flow_style=False,
        ),
        end="",
    )
    return 0


if __name__ == "__main__":
    configure_utf8_stdio()
    raise SystemExit(main())
