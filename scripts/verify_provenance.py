#!/usr/bin/env python3
"""Verify stored PR artifacts and optionally backfill missing merge SHAs."""

from __future__ import annotations

import argparse
import hashlib
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _wiki_root import WIKI_ROOT, configure_utf8_stdio  # noqa: E402


def verify_local_provenance(root: Path) -> list[str]:
    errors = []
    artifact_root = root / "artifacts" / "prs"
    for provenance_path in sorted(artifact_root.glob("*/PR-*/PROVENANCE.yaml")):
        relative = provenance_path.relative_to(root)
        try:
            provenance = yaml.safe_load(
                provenance_path.read_text(encoding="utf-8")
            ) or {}
        except (OSError, yaml.YAMLError) as error:
            errors.append(f"{relative}: cannot parse provenance: {error}")
            continue
        patch = provenance_path.parent / "diff.patch"
        if not patch.is_file():
            errors.append(f"{relative}: missing diff.patch")
            continue
        data = patch.read_bytes()
        observed_hash = hashlib.sha256(data).hexdigest()
        if provenance.get("diff_sha256") != observed_hash:
            errors.append(
                f"{relative}: diff_sha256 mismatch "
                f"({provenance.get('diff_sha256')} != {observed_hash})"
            )
        if int(provenance.get("diff_bytes") or -1) != len(data):
            errors.append(
                f"{relative}: diff_bytes mismatch "
                f"({provenance.get('diff_bytes')} != {len(data)})"
            )
        directory_pr = provenance_path.parent.name.removeprefix("PR-")
        if str(provenance.get("pr")) != directory_pr:
            errors.append(
                f"{relative}: PR number does not match artifact directory"
            )
        if not provenance.get("repo") or not provenance.get("url"):
            errors.append(f"{relative}: missing repo/url provenance")
    return errors


def _split_page(text: str) -> tuple[dict[str, Any], str]:
    match = re.match(r"^---\s*\r?\n(.*?)\r?\n---\s*\r?\n(.*)", text, re.DOTALL)
    if not match:
        raise ValueError("missing frontmatter")
    frontmatter = yaml.safe_load(match.group(1)) or {}
    return frontmatter, match.group(2)


def _gh_merge_sha(repo: str, pr: int) -> str | None:
    result = subprocess.run(
        ["gh", "api", f"repos/{repo}/pulls/{pr}", "--jq", ".merge_commit_sha"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip())
    sha = result.stdout.strip()
    return sha if re.fullmatch(r"[0-9a-f]{40}", sha) else None


def backfill_merge_shas(root: Path, dry_run: bool = False) -> tuple[int, list[str]]:
    changed = 0
    errors = []
    for page in sorted((root / "sources" / "prs").glob("*/*.md")):
        text = page.read_text(encoding="utf-8")
        try:
            frontmatter, body = _split_page(text)
        except (ValueError, yaml.YAMLError) as error:
            errors.append(f"{page.relative_to(root)}: {error}")
            continue
        if frontmatter.get("status") != "merged" or frontmatter.get("merge_sha"):
            continue
        repo = str(frontmatter.get("repo") or "")
        pr = int(frontmatter.get("pr") or 0)
        try:
            sha = _gh_merge_sha(repo, pr)
        except RuntimeError as error:
            errors.append(f"{page.relative_to(root)}: {error}")
            continue
        if not sha:
            errors.append(f"{page.relative_to(root)}: upstream has no merge SHA")
            continue
        frontmatter["merge_sha"] = sha
        if not dry_run:
            rendered = yaml.safe_dump(
                frontmatter,
                sort_keys=False,
                allow_unicode=True,
                default_flow_style=False,
            )
            page.write_text(f"---\n{rendered}---\n{body}", encoding="utf-8")
        changed += 1
    return changed, errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backfill-merge-shas", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    errors = verify_local_provenance(WIKI_ROOT)
    if args.backfill_merge_shas:
        changed, backfill_errors = backfill_merge_shas(
            WIKI_ROOT, dry_run=args.dry_run
        )
        print(f"merge_sha_backfilled={changed}")
        errors.extend(backfill_errors)
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print("OK: artifact provenance verified")
    return 0


if __name__ == "__main__":
    configure_utf8_stdio()
    raise SystemExit(main())
