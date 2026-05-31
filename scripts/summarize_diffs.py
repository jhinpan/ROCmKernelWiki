#!/usr/bin/env python3
"""Write a compact diff_summary.md into each artifacts/prs/<repo>/PR-<N>/ bundle so
an agent can learn the essence of a change without reading the full ~2800-word diff.

The summary captures: files touched (with +/- line counts), the dominant file
types/areas, and the first few non-trivial added lines per kernel-ish file (the
"what actually changed" signal). Idempotent.

Run:
  python3 scripts/summarize_diffs.py
  python3 scripts/summarize_diffs.py --dry-run
"""
import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _wiki_root import WIKI_ROOT  # noqa: E402

ART_DIR = WIKI_ROOT / "artifacts" / "prs"
KERNEL_EXT = (".cu", ".cuh", ".hip", ".cpp", ".hpp", ".h", ".inc", ".s", ".asm",
              ".py", ".mlir")


def parse_diff(text):
    """Return list of {file, added, removed, sample_adds}."""
    files = []
    cur = None
    for line in text.splitlines():
        m = re.match(r'^diff --git a/(\S+) b/(\S+)', line)
        if m:
            if cur:
                files.append(cur)
            cur = {"file": m.group(2), "added": 0, "removed": 0, "adds": []}
            continue
        if cur is None:
            continue
        if line.startswith("+") and not line.startswith("+++"):
            cur["added"] += 1
            body = line[1:].strip()
            if (len(cur["adds"]) < 4 and len(body) > 8
                    and not body.startswith(("//", "*", "/*", "#"))):
                cur["adds"].append(body[:120])
        elif line.startswith("-") and not line.startswith("---"):
            cur["removed"] += 1
    if cur:
        files.append(cur)
    return files


def summarize(files, truncated):
    total_add = sum(f["added"] for f in files)
    total_rem = sum(f["removed"] for f in files)
    kernel_files = [f for f in files if f["file"].lower().endswith(KERNEL_EXT)]
    lines = []
    lines.append(f"# Diff summary")
    lines.append("")
    lines.append(f"- **files changed:** {len(files)}"
                 + (" (diff was byte-capped; summary is partial)" if truncated else ""))
    lines.append(f"- **lines:** +{total_add} / -{total_rem}")
    lines.append(f"- **kernel-ish files:** {len(kernel_files)}")
    lines.append("")
    lines.append("## Files (by churn)")
    lines.append("")
    for f in sorted(files, key=lambda x: -(x["added"] + x["removed"]))[:15]:
        lines.append(f"- `{f['file']}`  (+{f['added']}/-{f['removed']})")
    lines.append("")
    shown = [f for f in kernel_files if f["adds"]][:5]
    if shown:
        lines.append("## Key added lines (kernel files)")
        lines.append("")
        for f in shown:
            lines.append(f"**`{f['file']}`**")
            lines.append("```")
            for a in f["adds"]:
                lines.append(a)
            lines.append("```")
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    diffs = sorted(ART_DIR.rglob("diff.patch"))
    n = 0
    for d in diffs:
        try:
            text = d.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        truncated = False
        prov = d.parent / "PROVENANCE.yaml"
        if prov.exists() and "truncated: true" in prov.read_text(encoding="utf-8"):
            truncated = True
        files = parse_diff(text)
        if not files:
            continue
        summary = summarize(files, truncated)
        if not args.dry_run:
            (d.parent / "diff_summary.md").write_text(summary, encoding="utf-8")
        n += 1
    print(f"{'would write' if args.dry_run else 'wrote'} {n} diff_summary.md files")


if __name__ == "__main__":
    main()
