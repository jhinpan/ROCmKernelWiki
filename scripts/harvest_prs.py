#!/usr/bin/env python3
"""Harvest merged PRs from tracked ROCm repositories into source PR pages.

Uses the GitHub CLI (`gh api graphql`) to page through merged PRs, captures
title / author / merge date / merge SHA / changed file paths, classifies each
PR include/defer/exclude per data/inclusion-policy.yaml, writes:

  candidates/<short>.yaml          per-repo candidate ledger (all PRs + decision)
  sources/prs/<short>/PR-<N>.md    one source page per INCLUDED PR

Run:
  python3 scripts/harvest_prs.py --repo composable_kernel --max 2710
  python3 scripts/harvest_prs.py --all          # all tracked repos
  python3 scripts/harvest_prs.py --all --dry-run

Requires: `gh auth status` authenticated. Network access required.
"""

import argparse
import json
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _wiki_root import WIKI_ROOT  # noqa: E402

POLICY = yaml.safe_load((WIKI_ROOT / "data" / "inclusion-policy.yaml").read_text())
CUTOFF = yaml.safe_load((WIKI_ROOT / "data" / "refresh-cutoff.yaml").read_text())["cutoff_date"]

GRAPHQL = """
query($owner:String!, $name:String!, $cursor:String) {
  repository(owner:$owner, name:$name) {
    pullRequests(states:MERGED, first:50, after:$cursor,
                 orderBy:{field:CREATED_AT, direction:DESC}) {
      pageInfo { hasNextPage endCursor }
      nodes {
        number title createdAt mergedAt
        author { login }
        mergeCommit { oid }
        bodyText
        files(first:60) { nodes { path } totalCount }
      }
    }
  }
}
"""


def gh_graphql(owner, name, cursor):
    args = ["gh", "api", "graphql", "-f", f"query={GRAPHQL}",
            "-F", f"owner={owner}", "-F", f"name={name}"]
    if cursor:
        args += ["-F", f"cursor={cursor}"]
    for attempt in range(5):
        res = subprocess.run(args, capture_output=True, text=True)
        if res.returncode == 0:
            return json.loads(res.stdout)
        if "RATE_LIMITED" in res.stderr or "rate limit" in res.stderr.lower():
            wait = 30 * (attempt + 1)
            print(f"  rate-limited; sleeping {wait}s", file=sys.stderr)
            time.sleep(wait)
            continue
        # transient
        time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"gh graphql failed for {owner}/{name}: {res.stderr[:400]}")


def slugify_paths(paths):
    return [p for p in paths]


def classify(title, body, paths, repo_cfg):
    """Return (decision, reason, tagset) where decision in include/defer/exclude."""
    t = (title or "").lower()
    b = (body or "").lower()
    blob = t + " " + b + " " + " ".join(paths).lower()

    for kw in POLICY["exclude_keywords"]:
        if kw in t:
            return "exclude", f"title matches exclude keyword '{kw}'", {}

    if repo_cfg.get("rocm_filter"):
        if not any(m in blob for m in POLICY["rocm_markers"]):
            return "exclude", "rocm_filter repo: no AMD/ROCm marker", {}

    # path-based inclusion
    kernel_exts = (".cu", ".cuh", ".hip", ".s", ".asm", ".inc")
    path_hit = None
    for p in paths:
        pl = p.lower()
        if pl.endswith(kernel_exts):
            path_hit = p
            break
        if any(tok in pl for tok in ("kernel", "mfma", "wmma", "gemm", "attention",
                                     "flash", "moe", "/asm/", "ck_tile", "tensile")):
            path_hit = p
            break
        if pl.endswith(".py") and any(tok in blob for tok in
                                      ("triton", "mfma", "tl.dot", "flydsl", "kernel")):
            path_hit = p
            break

    kw_hit = next((k for k in POLICY["include_keywords"] if k in blob), None)

    if path_hit or kw_hit:
        reason = []
        if path_hit:
            reason.append(f"kernel path '{path_hit}'")
        if kw_hit:
            reason.append(f"keyword '{kw_hit}'")
        return "include", "; ".join(reason), infer_tags(blob, repo_cfg)

    return "defer", "no kernel path or keyword signal", {}


def infer_tags(blob, repo_cfg):
    """Heuristically infer tag/feature/kernel/language vocab from text."""
    archs = []
    for gfx in ("gfx942", "gfx950", "gfx1201", "gfx90a", "gfx1100"):
        if gfx in blob:
            archs.append(gfx)
    for name, gfx in (("mi300", "gfx942"), ("mi325", "gfx942"),
                      ("mi350", "gfx950"), ("mi355", "gfx950"),
                      ("cdna3", "gfx942"), ("cdna4", "gfx950"),
                      ("r9700", "gfx1201"), ("rdna4", "gfx1201")):
        if name in blob and gfx not in archs:
            archs.append(gfx)
    if not archs:
        archs = ["gfx942"]  # default datacenter target

    hw = []
    for feat, keys in (
        ("mfma", ("mfma", "matrix core", "v_mfma")),
        ("wmma", ("wmma",)),
        ("lds", ("lds", "shared memory", "groupshared")),
        ("fp8", ("fp8", "e4m3", "e5m2", "fnuz")),
        ("fp4", ("fp4", "mxfp4")),
        ("mxfp", ("mxfp", "microscaling")),
        ("bf16", ("bf16", "bfloat16")),
        ("fp16", ("fp16", "half")),
        ("int8", ("int8",)),
        ("async-copy", ("async", "direct-to-lds", "global_load_lds")),
        ("block-scale", ("block scale", "block-scale", "blockscale")),
    ):
        if any(k in blob for k in keys):
            hw.append(feat)

    kt = []
    for kind, keys in (
        ("flash-attention", ("flash", "flashattention")),
        ("attention", ("attention",)),
        ("paged-attention", ("paged",)),
        ("mla", ("mla", "latent attention")),
        ("fused-moe", ("fused moe", "fused_moe")),
        ("moe", ("moe", "mixture of experts", "expert")),
        ("fp8-gemm", ("fp8 gemm", "fp8gemm")),
        ("grouped-gemm", ("grouped gemm", "grouped_gemm", "group gemm")),
        ("gemm", ("gemm", "matmul")),
        ("gemv", ("gemv",)),
        ("rmsnorm", ("rmsnorm",)),
        ("layernorm", ("layernorm", "layer norm")),
        ("softmax", ("softmax",)),
        ("rope", ("rope", "rotary")),
        ("kv-cache", ("kv cache", "kv_cache", "kvcache")),
        ("all-reduce", ("all reduce", "allreduce", "all_reduce")),
        ("quantization", ("quant", "dequant")),
    ):
        if any(k in blob for k in keys):
            kt.append(kind)

    langs = list(repo_cfg.get("default_languages", ["hip"]))
    if "triton" in blob and "triton" not in langs:
        langs.append("triton")

    tags = sorted(set(hw + kt + archs))
    return {
        "architectures": archs,
        "hardware_features": hw,
        "kernel_types": kt,
        "languages": langs,
        "tags": tags or archs,
    }


def write_pr_page(short, repo_full, node, tags, reason):
    num = node["number"]
    out_dir = WIKI_ROOT / "sources" / "prs" / short
    out_dir.mkdir(parents=True, exist_ok=True)
    merged = (node.get("mergedAt") or node.get("createdAt") or "")[:10]
    sha = (node.get("mergeCommit") or {}).get("oid", "")[:12]
    author = (node.get("author") or {}).get("login") or "unknown"
    title = (node.get("title") or "").replace('"', "'").strip()
    files = [f["path"] for f in (node.get("files") or {}).get("nodes", [])]
    total_files = (node.get("files") or {}).get("totalCount", len(files))
    changed = files[:20]

    fm = {
        "id": f"pr-{short}-{num}",
        "repo": repo_full,
        "pr": num,
        "title": title,
        "author": author,
        "date": merged,
        "url": f"https://github.com/{repo_full}/pull/{num}",
        "source_category": "upstream-code",
        "architectures": tags["architectures"],
        "tags": tags["tags"],
        "techniques": [],
        "hardware_features": tags["hardware_features"],
        "kernel_types": tags["kernel_types"],
        "languages": tags["languages"],
        "captured_at": CUTOFF,
        "status": "merged",
        "merge_sha": sha,
        "inclusion_reason": reason,
        "changed_paths": changed,
    }
    body_text = (node.get("bodyText") or "").strip()
    summary = body_text[:1200] if body_text else "_No PR description provided upstream._"

    yaml_fm = yaml.dump(fm, sort_keys=False, allow_unicode=True, default_flow_style=False)
    md = f"""---
{yaml_fm}---

# {title}

**Repository:** [{repo_full}]({fm['url'].rsplit('/pull',1)[0]}) · **PR:** [#{num}]({fm['url']}) · **Merged:** {merged} · **Author:** @{author}

**Inclusion reason:** {reason}

## Summary (from upstream PR description)

{summary}

## Changed files ({total_files} total, first {len(changed)} shown)

"""
    for p in changed:
        md += f"- `{p}`\n"
    md += f"""
## Provenance

- Merge commit: `{sha}`
- Captured at knowledge cutoff: {CUTOFF}
- Source of truth: the upstream PR linked above. This page summarizes upstream
  metadata; consult the PR for the authoritative diff.
"""
    (out_dir / f"PR-{num}.md").write_text(md, encoding="utf-8")


def harvest_repo(repo_cfg, max_prs, dry_run):
    repo_full = repo_cfg["repo"]
    short = repo_cfg["short"]
    owner, name = repo_full.split("/")
    print(f"\n=== {repo_full} (max {max_prs}) ===")
    cursor = None
    ledger = []
    n_seen = n_inc = 0
    while True:
        data = gh_graphql(owner, name, cursor)
        conn = data["data"]["repository"]["pullRequests"]
        for node in conn["nodes"]:
            n_seen += 1
            paths = [f["path"] for f in (node.get("files") or {}).get("nodes", [])]
            decision, reason, tags = classify(
                node.get("title"), node.get("bodyText"), paths, repo_cfg)
            ledger.append({
                "pr": node["number"],
                "title": (node.get("title") or "")[:120],
                "merged_at": (node.get("mergedAt") or "")[:10],
                "decision": decision,
                "reason": reason,
            })
            if decision == "include":
                n_inc += 1
                if not dry_run:
                    write_pr_page(short, repo_full, node, tags, reason)
            if n_seen >= max_prs:
                break
        if n_seen >= max_prs or not conn["pageInfo"]["hasNextPage"]:
            break
        cursor = conn["pageInfo"]["endCursor"]
        if n_seen % 200 == 0:
            print(f"  ...{n_seen} scanned, {n_inc} included")

    # write ledger
    if not dry_run:
        led_path = WIKI_ROOT / "candidates" / f"{short}.yaml"
        led_path.write_text(yaml.dump({
            "repo": repo_full,
            "scanned": n_seen,
            "included": n_inc,
            "deferred": sum(1 for x in ledger if x["decision"] == "defer"),
            "excluded": sum(1 for x in ledger if x["decision"] == "exclude"),
            "cutoff": CUTOFF,
            "prs": ledger,
        }, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(f"  DONE {repo_full}: scanned={n_seen} included={n_inc}")
    return n_seen, n_inc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", help="short name of a single tracked repo")
    ap.add_argument("--all", action="store_true", help="harvest all tracked repos")
    ap.add_argument("--max", type=int, default=100000, help="max PRs to scan per repo")
    ap.add_argument("--dry-run", action="store_true", help="classify only, write nothing")
    args = ap.parse_args()

    repos = POLICY["tracked_repos"]
    if args.repo:
        repos = [r for r in repos if r["short"] == args.repo]
        if not repos:
            print(f"unknown repo '{args.repo}'", file=sys.stderr)
            sys.exit(1)
    elif not args.all:
        ap.error("specify --repo <short> or --all")

    grand_seen = grand_inc = 0
    for rc in repos:
        s, i = harvest_repo(rc, args.max, args.dry_run)
        grand_seen += s
        grand_inc += i
    print(f"\n=== TOTAL: scanned={grand_seen} included={grand_inc} ===")


if __name__ == "__main__":
    main()
