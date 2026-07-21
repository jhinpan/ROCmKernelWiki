#!/usr/bin/env python3
"""Text search across wiki bodies, source pages, and artifact bundles.

Usage:
    grep_wiki.py "v_mfma_f32_16x16x16"
    grep_wiki.py "buffer_load" --only wiki
    grep_wiki.py "s_waitcnt" --context 3
    grep_wiki.py "mfma async" --any
    grep_wiki.py "ds_swizzle" --only artifacts
"""

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _wiki_root import WIKI_ROOT, configure_utf8_stdio  # noqa: E402

CODE_EXTS = {
    ".md", ".cu", ".cuh", ".hip", ".s", ".asm", ".inc",
    ".cpp", ".cxx", ".cc", ".c", ".h", ".hpp", ".hxx", ".inl",
    ".py", ".mlir", ".patch", ".txt", ".sh", ".yaml", ".yml", ".json",
}


def iter_files(scope, exts=None):
    dirs = {
        "wiki": ["wiki"],
        "sources": ["sources"],
        "all": ["wiki", "sources"],
        "artifacts": ["artifacts"],
    }
    sub_list = dirs.get(scope, ["wiki", "sources"])
    if exts and "artifacts" not in sub_list and scope not in ("wiki", "sources"):
        sub_list = sub_list + ["artifacts"]
    if scope == "artifacts" and not exts:
        search_exts = set(CODE_EXTS)
    else:
        search_exts = {".md"} | (exts or set())
    for sub in sub_list:
        base = WIKI_ROOT / sub
        if not base.exists():
            continue
        for f in base.rglob("*"):
            if f.is_file() and f.suffix.lower() in search_exts:
                yield f


def grep_file(path, patterns, context, any_match):
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return []
    results = []
    for i, line in enumerate(lines):
        matched = (any if any_match else all)(p.search(line) for p in patterns)
        if matched:
            lo = max(0, i - context)
            hi = min(len(lines), i + context + 1)
            block = []
            for j in range(lo, hi):
                marker = ">>" if j == i else "  "
                block.append(f"{marker} {j+1}: {lines[j]}")
            results.append((i + 1, "\n".join(block)))
    return results


def main():
    parser = argparse.ArgumentParser(description="Regex search across the ROCm kernel wiki")
    parser.add_argument("patterns", nargs="+", help="One or more regex patterns")
    parser.add_argument("--only", choices=["wiki", "sources", "all", "artifacts"],
                        default="all", help="Restrict scope (default all = wiki+sources)")
    parser.add_argument("--ext", action="append", default=[],
                        help="Additional file extensions to search (e.g. --ext .s --ext .cu)")
    parser.add_argument("--any", action="store_true", help="Match if ANY pattern matches (default ALL)")
    parser.add_argument("--context", type=int, default=0, help="Context lines around each match")
    parser.add_argument("--ignore-case", "-i", action="store_true")
    parser.add_argument("--limit", type=int, default=200, help="Max matching files to print")
    args = parser.parse_args()

    flags = re.IGNORECASE if args.ignore_case else 0
    patterns = [re.compile(p, flags) for p in args.patterns]
    exts = {e if e.startswith(".") else "." + e for e in args.ext} or None

    total = 0
    files_with_hits = 0
    for f in iter_files(args.only, exts):
        hits = grep_file(f, patterns, args.context, args.any)
        if not hits:
            continue
        files_with_hits += 1
        if files_with_hits > args.limit:
            print(f"... (truncated at {args.limit} files)")
            break
        print(f"\n=== {f.relative_to(WIKI_ROOT)} ({len(hits)} match) ===")
        for _, block in hits:
            print(block)
            if args.context:
                print("--")
            total += 1
    if files_with_hits == 0:
        print("No matches.")
    else:
        print(f"\n# {total} match(es) across {files_with_hits} file(s)")


if __name__ == "__main__":
    configure_utf8_stdio()
    main()
