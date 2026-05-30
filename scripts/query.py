#!/usr/bin/env python3
"""Unified query tool for the ROCm kernel wiki.

Supports natural-language keyword queries, tag filters, repo filters, and type filters.

Usage:
    query.py "how to pipeline MFMA on MI300"
    query.py --tag mfma --type kernel
    query.py --repo composable_kernel --limit 20
    query.py --language gcn-asm
    query.py --symptom bank-conflicts

Returns a ranked list of matching pages with titles, paths, and key frontmatter fields.
"""

import argparse
import re
import sys
import yaml
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _wiki_root import WIKI_ROOT  # noqa: E402


_ALIAS_CACHE = None


def load_alias_expansions():
    """Return a dict mapping lowercased alias → canonical term, from data/aliases.yaml."""
    global _ALIAS_CACHE
    if _ALIAS_CACHE is not None:
        return _ALIAS_CACHE
    out = {}
    aliases_path = WIKI_ROOT / "data" / "aliases.yaml"
    try:
        raw = yaml.safe_load(aliases_path.read_text(encoding="utf-8")) or {}
    except Exception:
        _ALIAS_CACHE = {}
        return _ALIAS_CACHE
    for canonical, variants in raw.items():
        if not isinstance(canonical, str):
            continue
        out.setdefault(canonical.lower(), canonical)
        for v in (variants or []):
            if isinstance(v, str):
                out.setdefault(v.lower(), canonical)
    _ALIAS_CACHE = out
    return out


def expand_keyword(kw):
    """Return search variants for a keyword: original plus any canonical term."""
    aliases = load_alias_expansions()
    canonical = aliases.get(kw.lower())
    if canonical and canonical.lower() != kw.lower():
        return [kw, canonical]
    return [kw]


def load_frontmatter(path):
    """Parse YAML frontmatter. Returns (fm_dict, body_str) or (None, None)."""
    try:
        content = path.read_text(encoding="utf-8")
    except Exception:
        return None, None
    m = re.match(r'^---\s*\r?\n(.*?)\r?\n---\s*\r?\n(.*)', content, re.DOTALL)
    if not m:
        return None, None
    try:
        fm = yaml.safe_load(m.group(1))
        if not isinstance(fm, dict):
            return None, None
        return fm, m.group(2)
    except yaml.YAMLError:
        return None, None


def load_all_pages():
    """Load frontmatter + body for every sources/*.md and wiki/*.md file."""
    pages = []
    for subdir in ["sources", "wiki"]:
        base = WIKI_ROOT / subdir
        if not base.exists():
            continue
        for md in base.rglob("*.md"):
            fm, body = load_frontmatter(md)
            if fm is None:
                continue
            pages.append({
                "path": str(md.relative_to(WIKI_ROOT)),
                "fm": fm,
                "body": body or "",
            })
    return pages


def detect_page_type(fm, path):
    """Return a page-type label for filtering."""
    if "type" in fm:
        return f"wiki-{fm['type']}"
    parts = path.split("/")
    if parts[0] == "sources" and len(parts) > 1:
        return f"source-{parts[1].rstrip('s')}"  # prs → source-pr, refs → source-ref
    return "unknown"


def score_keyword_match(fm, body, keywords):
    """Score a page by keyword matches in title, tags, and body (alias-aware)."""
    score = 0
    title_text = str(fm.get("title", "")).lower()
    tag_text = " ".join(
        str(v) for k in ("tags", "techniques", "hardware_features", "kernel_types",
                          "languages", "aliases", "symptoms")
        for v in (fm.get(k) or [])
    ).lower()
    body_lower = body.lower()
    for kw in keywords:
        best = 0
        for variant in expand_keyword(kw):
            v_l = variant.lower()
            vs = 0
            if v_l in title_text:
                vs += 10
            if v_l in tag_text:
                vs += 5
            vs += min(body_lower.count(v_l), 3)
            best = max(best, vs)
        score += best
    return score


def filter_pages(pages, args):
    out = []
    for p in pages:
        fm = p["fm"]
        path = p["path"]
        ptype = detect_page_type(fm, path)
        p["_ptype"] = ptype

        if args.type:
            if not ptype.endswith(args.type) and ptype != args.type:
                continue

        if args.tag:
            all_tags = set()
            for k in ("tags", "techniques", "hardware_features", "kernel_types", "languages"):
                all_tags.update(fm.get(k) or [])
            tag_variants = {v.lower() for v in expand_keyword(args.tag)}
            if not any(str(t).lower() in tag_variants for t in all_tags):
                continue

        if args.repo:
            repo = str(fm.get("repo", "")).lower()
            if args.repo.lower() not in repo:
                continue

        if args.language:
            langs = set(fm.get("languages") or [])
            tags = set(fm.get("tags") or [])
            if args.language not in langs and args.language not in tags:
                continue

        if args.architecture:
            archs = {str(a).lower() for a in (fm.get("architectures") or [])}
            arch_variants = {v.lower() for v in expand_keyword(args.architecture)}
            if not (archs & arch_variants):
                continue

        if args.symptom:
            symptoms = set(fm.get("symptoms") or [])
            if args.symptom not in symptoms:
                continue

        if args.confidence:
            if str(fm.get("confidence", "")) != args.confidence:
                continue

        out.append(p)
    return out


def format_result(page, compact=False):
    fm = page["fm"]
    title = fm.get("title", "Untitled")
    path = page["path"]
    pid = fm.get("id", "")
    ptype = page.get("_ptype", "?")

    if compact:
        return f"  [{ptype}] {pid}: {title}  ({path})"

    lines = [f"## {title}"]
    lines.append(f"- **id**: `{pid}`")
    lines.append(f"- **type**: `{ptype}`")
    lines.append(f"- **path**: `{path}`")
    if "architectures" in fm:
        lines.append(f"- **architectures**: {fm['architectures']}")
    for k in ("confidence", "reproducibility"):
        if k in fm:
            lines.append(f"- **{k}**: {fm[k]}")
    for k in ("tags", "techniques", "hardware_features", "kernel_types", "languages"):
        v = fm.get(k)
        if v:
            lines.append(f"- **{k}**: {v}")
    if "performance_claims" in fm and isinstance(fm["performance_claims"], list):
        for claim in fm["performance_claims"][:2]:
            lines.append(f"- **perf**: {claim.get('value')} {claim.get('metric')} on "
                         f"{claim.get('gpu')} ({claim.get('dtype')}, {claim.get('shape')})")
    if "sources" in fm:
        lines.append(f"- **sources**: {fm['sources'][:5]}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Query the ROCm kernel wiki")
    parser.add_argument("query", nargs="*", help="Free-text keywords")
    parser.add_argument("--type", help="Page type (kernel, technique, hardware, pattern, language, migration, pr, doc, blog, ref)")
    parser.add_argument("--tag", help="Filter by tag (tags/techniques/hardware_features/kernel_types/languages); alias-aware")
    parser.add_argument("--repo", help="Filter by source repo (partial, e.g. 'composable_kernel')")
    parser.add_argument("--language", help="Filter by language/DSL (hip, gcn-asm, composable-kernel, flydsl, triton, ...)")
    parser.add_argument("--architecture", help="Filter by architecture (gfx942, gfx950, gfx1201, ...); alias-aware")
    parser.add_argument("--symptom", help="Filter by pattern symptom (bank-conflicts, low-occupancy, ...)")
    parser.add_argument("--confidence", help="Filter by confidence (verified, source-reported, inferred, experimental)")
    parser.add_argument("--limit", type=int, default=10, help="Max results (default 10)")
    parser.add_argument("--compact", action="store_true", help="Compact one-line-per-result output")
    parser.add_argument("--paths-only", action="store_true", help="Output only file paths")
    args = parser.parse_args()

    pages = load_all_pages()
    pages = filter_pages(pages, args)

    keywords = []
    for q in args.query:
        for tok in re.split(r"\s+", q.strip()):
            if tok:
                keywords.append(tok)
    if keywords:
        for p in pages:
            p["_score"] = score_keyword_match(p["fm"], p["body"], keywords)
        pages = [p for p in pages if p["_score"] > 0]
        pages.sort(key=lambda x: (-x["_score"], x["path"]))
    else:
        pages.sort(key=lambda x: x["path"])

    pages = pages[:args.limit]

    if args.paths_only:
        for p in pages:
            print(p["path"])
        return

    if not pages:
        print("No matching pages.")
        return

    print(f"# {len(pages)} result(s)")
    print()
    for p in pages:
        print(format_result(p, compact=args.compact))
        if not args.compact:
            print()


if __name__ == "__main__":
    main()
