#!/usr/bin/env python3
"""Retrieve a single wiki page by its id or path.

Usage:
    get_page.py kernel-ck-fp8-gemm           # by id
    get_page.py pr-composable_kernel-1234     # by id
    get_page.py wiki/kernels/ck-fp8-gemm.md   # by path
    get_page.py kernel-ck-fp8-gemm --body-only
    get_page.py kernel-ck-fp8-gemm --frontmatter-only
    get_page.py kernel-ck-fp8-gemm --follow-sources
    get_page.py kernel-ck-fp8-gemm --include-code
"""

import argparse
import re
import sys
import yaml
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _wiki_root import WIKI_ROOT  # noqa: E402


def find_page(lookup):
    """Find a page by id or relative path. Returns Path or None."""
    if "/" in lookup or lookup.endswith(".md"):
        p = WIKI_ROOT / lookup
        return p if p.exists() else None
    for subdir in ["wiki", "sources"]:
        base = WIKI_ROOT / subdir
        if not base.exists():
            continue
        for md in base.rglob("*.md"):
            try:
                content = md.read_text(encoding="utf-8")
            except Exception:
                continue
            m = re.match(r'^---\s*\r?\n(.*?)\r?\n---', content, re.DOTALL)
            if not m:
                continue
            try:
                fm = yaml.safe_load(m.group(1))
            except yaml.YAMLError:
                continue
            if isinstance(fm, dict) and fm.get("id") == lookup:
                return md
    return None


def split_frontmatter(content):
    m = re.match(r'^---\s*\r?\n(.*?)\r?\n---\s*\r?\n(.*)', content, re.DOTALL)
    if not m:
        return None, content
    try:
        fm = yaml.safe_load(m.group(1))
    except yaml.YAMLError:
        fm = None
    return fm, m.group(2)


def _resolve_artifact_dir(page_path, fm):
    """Return (rel_str, abs_path, is_fallback) or (None, None, False)."""
    fm = fm or {}
    if fm.get("artifact_dir"):
        rel = str(fm["artifact_dir"])
        return rel, WIKI_ROOT / rel, False
    try:
        rel_page = page_path.relative_to(WIKI_ROOT)
    except ValueError:
        return None, None, False
    parts = rel_page.parts
    if len(parts) < 2 or parts[0] != "sources":
        return None, None, False
    candidate = None
    if parts[1] == "blogs" and len(parts) == 3:
        candidate = Path("artifacts") / "blogs" / page_path.stem
    elif parts[1] == "prs" and len(parts) == 4:
        candidate = Path("artifacts") / "prs" / parts[2] / page_path.stem
    if candidate is None:
        return None, None, False
    abs_path = WIKI_ROOT / candidate
    if not abs_path.is_dir():
        return None, None, False
    return str(candidate), abs_path, True


def main():
    parser = argparse.ArgumentParser(description="Get a wiki page by id or path")
    parser.add_argument("lookup", help="Page id or relative path")
    parser.add_argument("--body-only", action="store_true")
    parser.add_argument("--frontmatter-only", action="store_true")
    parser.add_argument("--follow-sources", action="store_true",
                        help="Also print a 500-char excerpt from each cited source")
    parser.add_argument("--include-code", action="store_true",
                        help="After the body, print files under the page's artifact_dir")
    args = parser.parse_args()

    page_path = find_page(args.lookup)
    if not page_path:
        print(f"ERROR: No page found for '{args.lookup}'", file=sys.stderr)
        sys.exit(1)

    content = page_path.read_text(encoding="utf-8")
    fm, body = split_frontmatter(content)

    if args.frontmatter_only:
        if fm:
            print(yaml.dump(fm, allow_unicode=True, sort_keys=False))
        return
    if args.body_only:
        print(body)
        return

    print(f"# {page_path.relative_to(WIKI_ROOT)}")
    print()
    print(content)

    if args.follow_sources and fm and "sources" in fm:
        print()
        print("---")
        print("## Cited Sources (excerpts)")
        print()
        for src_id in fm.get("sources", []):
            src_page = find_page(src_id)
            if src_page:
                _, src_body = split_frontmatter(src_page.read_text(encoding="utf-8"))
                excerpt = (src_body or "")[:500].strip()
                print(f"### {src_id}")
                print(f"`{src_page.relative_to(WIKI_ROOT)}`")
                print()
                print(excerpt)
                print()

    if args.include_code:
        ad, ad_path, is_fallback = _resolve_artifact_dir(page_path, fm)
        if ad_path and ad_path.is_dir():
            exts = {".cu", ".cuh", ".hip", ".s", ".asm", ".inc", ".cpp", ".cxx",
                    ".cc", ".c", ".h", ".hpp", ".hxx", ".inl", ".py", ".mlir",
                    ".patch", ".sh", ".md", ".yaml", ".yml", ".txt", ".json"}
            print()
            print("---")
            suffix = " (conventional path — artifact_dir not backfilled)" if is_fallback else ""
            print(f"## Artifact Bundle: `{ad}`{suffix}")
            print()
            for f in sorted(ad_path.rglob("*")):
                if not f.is_file() or f.suffix.lower() not in exts:
                    continue
                print(f"### `{f.relative_to(ad_path)}`")
                print()
                try:
                    data = f.read_bytes()
                    if len(data) > 200 * 1024:
                        print(f"*(file is {len(data)} bytes; showing first 200 KiB)*")
                        data = data[:200 * 1024]
                    print("```" + (f.suffix.lstrip(".") or ""))
                    print(data.decode("utf-8", errors="replace"))
                    print("```")
                    print()
                except Exception as e:
                    print(f"*(could not read: {e})*")
                    print()
        elif ad is not None:
            print()
            print(f"*(artifact_dir '{ad}' not found on disk)*")


if __name__ == "__main__":
    main()
