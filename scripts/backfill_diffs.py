#!/usr/bin/env python3
"""Backfill real PR diffs into artifacts/ for the highest-signal kernel PRs.

Ranks every INCLUDE-classified PR page by how strong a kernel-learning signal it
carries (kernel file paths in changed_paths + topic keywords + AMD-native repo),
fetches the unified diff for the top N via the GitHub API, and writes:

  artifacts/prs/<repo>/PR-<N>/diff.patch     (size-capped, real upstream diff)
  artifacts/prs/<repo>/PR-<N>/PROVENANCE.yaml (repo, pr, merge_sha, url, sha256, bytes)

It also stamps `artifact_dir: artifacts/prs/<repo>/PR-<N>` onto the PR page so
get_page.py --include-code and query.py surface the bundle.

Run:
  python3 scripts/backfill_diffs.py --top 1000
  python3 scripts/backfill_diffs.py --top 1000 --dry-run   # rank only, fetch nothing
"""
import argparse
import hashlib
import re
import subprocess
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _wiki_root import WIKI_ROOT  # noqa: E402

PRS_DIR = WIKI_ROOT / "sources" / "prs"
ART_DIR = WIKI_ROOT / "artifacts" / "prs"
DIFF_CAP = 256 * 1024  # 256 KiB per diff

# AMD-native repos carry the strongest learning signal; weight them up.
NATIVE_REPOS = {"composable_kernel", "aiter", "hipBLASLt", "Tensile",
                "flash-attention", "FlyDSL", "triton", "rocBLAS"}
KERNEL_EXTS = (".cu", ".cuh", ".hip", ".s", ".asm", ".inc", ".cpp", ".hpp", ".py")
TOPIC_KW = ("mfma", "wmma", "gemm", "attention", "flash", "fmha", "moe", "fp8",
            "fp4", "mxfp", "gfx950", "gfx942", "kernel", "lds", "async", "quant",
            "rmsnorm", "layernorm", "rope", "paged", "decode", "prefill")
HOT_TAGS = {"mfma", "wmma", "fp8", "fp4", "mxfp", "block-scale", "matrix-core"}


def split_fm(text):
    m = re.match(r'^---\s*\r?\n(.*?)\r?\n---\s*\r?\n(.*)', text, re.DOTALL)
    if not m:
        return None, text
    try:
        return yaml.safe_load(m.group(1)), m.group(2)
    except yaml.YAMLError:
        return None, text


def score(fm):
    s = 0
    repo_short = str(fm.get("repo", "")).split("/")[-1]
    if repo_short in NATIVE_REPOS:
        s += 30
    paths = fm.get("changed_paths") or []
    kpaths = [p for p in paths if str(p).lower().endswith(KERNEL_EXTS)]
    s += min(len(kpaths), 6) * 8
    for p in paths:
        pl = str(p).lower()
        if any(t in pl for t in ("mfma", "wmma", "gemm", "attention", "fmha",
                                 "moe", "/asm/", "ck_tile", "kernel")):
            s += 4
    blob = (str(fm.get("title", "")) + " " + " ".join(str(p) for p in paths)).lower()
    s += sum(2 for kw in TOPIC_KW if kw in blob)
    s += len(set(fm.get("hardware_features") or []) & HOT_TAGS) * 5
    s += len(fm.get("kernel_types") or []) * 2
    # fewer files = more focused/readable diff
    if 1 <= len(paths) <= 8:
        s += 6
    return s


def gh_diff(repo_full, pr):
    """Fetch the unified diff for a PR via the GitHub diff media type."""
    for attempt in range(4):
        res = subprocess.run(
            ["gh", "api", f"repos/{repo_full}/pulls/{pr}",
             "-H", "Accept: application/vnd.github.v3.diff"],
            capture_output=True, text=True)
        if res.returncode == 0:
            return res.stdout
        err = res.stderr.lower()
        if "rate limit" in err or "was submitted too quickly" in err or "abuse" in err:
            wait = 30 * (attempt + 1)
            print(f"    rate-limited; sleeping {wait}s", file=sys.stderr)
            time.sleep(wait)
            continue
        if "not found" in err or "no commit found" in err:
            return None
        time.sleep(4 * (attempt + 1))
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=1000)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    # rank
    ranked = []
    for md in PRS_DIR.rglob("PR-*.md"):
        fm, _ = split_fm(md.read_text(encoding="utf-8"))
        if not fm:
            continue
        ranked.append((score(fm), md, fm))
    ranked.sort(key=lambda x: (-x[0], str(x[1])))
    top = ranked[:args.top]
    print(f"ranked {len(ranked)} PR pages; taking top {len(top)} "
          f"(score range {top[-1][0]}..{top[0][0]})")

    by_repo = {}
    for sc, md, fm in top:
        by_repo.setdefault(str(fm.get("repo")), 0)
        by_repo[str(fm.get("repo"))] += 1
    for r, c in sorted(by_repo.items(), key=lambda x: -x[1]):
        print(f"  {c:4d}  {r}")
    if args.dry_run:
        return

    ok = skip = fail = 0
    for i, (sc, md, fm) in enumerate(top, 1):
        repo_full = str(fm["repo"])
        repo_short = repo_full.split("/")[-1]
        pr = fm["pr"]
        bundle = ART_DIR / repo_short / f"PR-{pr}"
        patch_path = bundle / "diff.patch"
        if patch_path.exists():
            skip += 1
        else:
            diff = gh_diff(repo_full, pr)
            if not diff:
                fail += 1
                continue
            data = diff.encode("utf-8", errors="replace")
            truncated = False
            if len(data) > DIFF_CAP:
                data = data[:DIFF_CAP]
                truncated = True
            bundle.mkdir(parents=True, exist_ok=True)
            patch_path.write_bytes(data)
            prov = {
                "repo": repo_full,
                "pr": pr,
                "url": fm.get("url"),
                "merge_sha": fm.get("merge_sha", ""),
                "captured_at": fm.get("captured_at", ""),
                "diff_bytes": len(data),
                "diff_sha256": hashlib.sha256(data).hexdigest(),
                "truncated": truncated,
                "note": ("Verbatim upstream PR diff fetched via the GitHub API. "
                         "Authoritative source is the upstream PR; this is a "
                         "byte-capped snapshot for offline study."),
            }
            (bundle / "PROVENANCE.yaml").write_text(
                yaml.dump(prov, sort_keys=False, allow_unicode=True), encoding="utf-8")
            ok += 1

        # stamp artifact_dir onto the page if absent
        text = md.read_text(encoding="utf-8")
        if "artifact_dir:" not in text:
            rel = f"artifacts/prs/{repo_short}/PR-{pr}"
            text = text.replace(
                "\nstatus: merged",
                f"\nartifact_dir: {rel}\nstatus: merged", 1)
            md.write_text(text, encoding="utf-8")

        if i % 100 == 0:
            print(f"  [{i}/{len(top)}] ok={ok} skip={skip} fail={fail}")

    print(f"DONE: fetched={ok} already-present={skip} failed={fail}")


if __name__ == "__main__":
    main()
