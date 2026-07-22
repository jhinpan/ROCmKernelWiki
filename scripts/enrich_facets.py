#!/usr/bin/env python3
"""Enrich PR-page facets (techniques / hardware_features / kernel_types) by mining
each PR's changed_paths, title, body, and — when present — the real fetched diff in
artifacts/prs/<repo>/PR-<N>/diff.patch.

Measured before this script: techniques inferred on 0% of PR pages,
hardware_features empty on 56%, kernel_types empty on 34%. This restores those
filter axes so `query.py --tag mfma-pipelining` etc. actually return PR evidence.

All inferred values are drawn ONLY from data/tags.yaml controlled vocabulary, so
the result still passes validate.py. A `facet_source: inferred` marker records that
these were machine-derived. Re-runnable / idempotent.

Run:
  python3 scripts/enrich_facets.py            # enrich all PR pages
  python3 scripts/enrich_facets.py --dry-run  # report coverage delta only
"""
import argparse
import re
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _wiki_root import WIKI_ROOT  # noqa: E402

PRS_DIR = WIKI_ROOT / "sources" / "prs"
ART_DIR = WIKI_ROOT / "artifacts" / "prs"
TAGS = yaml.safe_load((WIKI_ROOT / "data" / "tags.yaml").read_text())
ALIASES = yaml.safe_load((WIKI_ROOT / "data" / "aliases.yaml").read_text()) or {}
VOCAB_HW = set(TAGS["hardware_features"])
VOCAB_TECH = set(TAGS["techniques"])
VOCAB_KT = set(TAGS["kernel_types"])
VOCAB_ARCH = set(TAGS["architectures"])

# keyword -> canonical vocab term. Keys are matched as lowercase substrings against
# (title + body + changed_paths + diff text).
HW_RULES = {
    "mfma": "mfma", "v_mfma": "mfma", "xdlop": "mfma", "matrix core": "matrix-core",
    "mfma_": "mfma", "wmma": "wmma", "v_wmma": "wmma",
    "lds": "lds", "__shared__": "lds", "ds_read": "ds-instructions",
    "ds_write": "ds-instructions", "ds_swizzle": "swizzle", "ds_permute": "permute",
    "ds_bpermute": "permute", "buffer_load": "buffer-instructions",
    "buffer_store": "buffer-instructions", "global_load": "global-instructions",
    "global_store": "global-instructions", "global_load_lds": "async-copy",
    "async": "async-copy", "s_waitcnt": "s-waitcnt", "vmcnt": "s-waitcnt",
    "agpr": "agpr", "dpp": "dpp", "permlane": "permute",
    "fp8": "fp8", "e4m3": "fp8", "e5m2": "fp8", "fnuz": "fp8", "f8": "fp8",
    "fp6": "fp6", "fp4": "fp4", "mxfp": "mxfp", "f8f6f4": "mxfp",
    "microscaling": "mxfp", "block scale": "block-scale", "blockscale": "block-scale",
    "bf16": "bf16", "bfloat16": "bf16", "fp16": "fp16", "half": "fp16",
    "int8": "int8", "wave64": "wave64", "wave32": "wave32", "wavefront": "wave64",
}
TECH_RULES = {
    "double buffer": "lds-double-buffering", "double-buffer": "lds-double-buffering",
    "ping-pong": "lds-double-buffering", "pingpong": "lds-double-buffering",
    "pipeline": "software-pipelining", "num_stages": "software-pipelining",
    "prefetch": "software-pipelining", "swizzl": "lds-swizzling",
    "bank conflict": "bank-conflict-avoidance", "bank-conflict": "bank-conflict-avoidance",
    "vectoriz": "vectorized-loads", "float4": "vectorized-loads",
    "dwordx4": "vectorized-loads", "nontemporal": "nontemporal-loads",
    "non-temporal": "nontemporal-loads", "oob": "buffer-oob-guard",
    "out-of-bound": "buffer-oob-guard", "reduce": "wave-reduce", "reduction": "wave-reduce",
    "occupancy": "occupancy-tuning", "waves_per_eu": "occupancy-tuning",
    "vgpr": "vgpr-budgeting", "register": "register-blocking",
    "split-k": "split-k", "splitk": "split-k", "stream-k": "stream-k", "streamk": "stream-k",
    "preshuffle": "preshuffle-layout", "quant": "fine-grained-quantization",
    "fusion": "kernel-fusion", "fused": "kernel-fusion", "fuse": "kernel-fusion",
    "persistent": "persistent-kernel", "epilogue": "epilogue-fusion",
    "mfma": "mfma-pipelining", "unroll": "loop-unrolling", "sched_barrier": "mfma-pipelining",
}
KT_RULES = {
    "hgemm": "hgemm", "sgemm": "sgemm", "fp8 gemm": "fp8-gemm", "fp8_gemm": "fp8-gemm",
    "grouped gemm": "grouped-gemm", "grouped_gemm": "grouped-gemm", "group_gemm": "grouped-gemm",
    "batched gemm": "batched-gemm", "gemm": "gemm", "matmul": "gemm", "gemv": "gemv",
    "flash": "flash-attention", "fmha": "flash-attention", "paged": "paged-attention",
    "mla": "mla", "attention": "attention", "fused_moe": "fused-moe", "fused moe": "fused-moe",
    "moe": "moe", "expert": "moe", "rmsnorm": "rmsnorm", "layernorm": "layernorm",
    "softmax": "softmax", "rope": "rope", "rotary": "rope", "kv cache": "kv-cache",
    "kv_cache": "kv-cache", "kvcache": "kv-cache", "all_reduce": "all-reduce",
    "allreduce": "all-reduce", "transpose": "transpose", "elementwise": "elementwise",
    "dequant": "dequantization", "quant": "quantization", "decode": "decode",
    "prefill": "prefill", "convolution": "convolution", "conv": "convolution",
}
ARCH_RULES = {
    str(alias).lower(): architecture
    for architecture in sorted(VOCAB_ARCH)
    for alias in [architecture, *(ALIASES.get(architecture) or [])]
}


def split_fm(text):
    m = re.match(r'^---\s*\r?\n(.*?)\r?\n---\s*\r?\n(.*)', text, re.DOTALL)
    if not m:
        return None, None, None
    try:
        fm = yaml.safe_load(m.group(1))
    except yaml.YAMLError:
        return None, None, None
    return fm, m.group(2), m.end(1)


def apply_rules(blob, rules, vocab):
    found = []
    for kw, term in rules.items():
        if term in vocab and kw in blob and term not in found:
            found.append(term)
    return found


def apply_architecture_rules(blob):
    found = []
    for keyword, architecture in ARCH_RULES.items():
        pattern = rf"(?<![a-z0-9]){re.escape(keyword)}(?![a-z0-9])"
        if re.search(pattern, blob) and architecture not in found:
            found.append(architecture)
    return found


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--diff-bytes", type=int, default=20000,
                    help="max diff bytes to scan per PR (default 20k)")
    args = ap.parse_args()

    pages = sorted(PRS_DIR.rglob("PR-*.md"))
    before = {"hw": 0, "tech": 0, "kt": 0}
    after = {"hw": 0, "tech": 0, "kt": 0}
    changed = 0

    for md in pages:
        text = md.read_text(encoding="utf-8")
        fm, body, _ = split_fm(text)
        if not fm:
            continue
        for key, cur in (("hw", "hardware_features"), ("tech", "techniques"),
                         ("kt", "kernel_types")):
            if fm.get(cur):
                before[key] += 1

        # Skip facet inference for non-kernel PRs whose paths merely happen to
        # contain words like "gemm" or "moe". Architecture inference remains
        # mandatory because it controls whether a PR enters the active corpus.
        title_l = str(fm.get("title", "")).lower()
        skip_facet_inference = bool(
            re.search(
                r"copyright|chore|\bdocs?\b|readme|changelog|bump|"
                r"\bci\b|lint|pre-commit|typo|comment|license|"
                r"clang-format|formatting|whitespace",
                title_l,
            )
        )

        repo_short = str(fm.get("repo", "")).split("/")[-1]
        pr = fm.get("pr")
        diff_path = ART_DIR / repo_short / f"PR-{pr}" / "diff.patch"
        diff_text = ""
        if diff_path.exists():
            try:
                diff_text = diff_path.read_bytes()[:args.diff_bytes].decode(
                    "utf-8", errors="replace")
            except Exception:
                diff_text = ""

        blob = " ".join([
            str(fm.get("title", "")),
            " ".join(str(p) for p in (fm.get("changed_paths") or [])),
            (body or "")[:2000],
            diff_text,
        ]).lower()

        if skip_facet_inference:
            new_hw = list(fm.get("hardware_features") or [])
            new_tech = list(fm.get("techniques") or [])
            new_kt = list(fm.get("kernel_types") or [])
        else:
            new_hw = sorted(set((fm.get("hardware_features") or [])
                                + apply_rules(blob, HW_RULES, VOCAB_HW)))
            new_tech = sorted(set((fm.get("techniques") or [])
                                  + apply_rules(blob, TECH_RULES, VOCAB_TECH)))
            new_kt = sorted(set((fm.get("kernel_types") or [])
                                + apply_rules(blob, KT_RULES, VOCAB_KT)))
        new_arch = sorted(set((fm.get("architectures") or [])
                              + apply_architecture_rules(blob)))

        # keep tags coherent: union of facets (validator allows any vocab value)
        new_tags = sorted(set((fm.get("tags") or []) + new_hw + new_kt + new_arch))

        dirty = (new_hw != (fm.get("hardware_features") or [])
                 or new_tech != (fm.get("techniques") or [])
                 or new_kt != (fm.get("kernel_types") or [])
                 or new_arch != (fm.get("architectures") or [])
                 or new_tags != (fm.get("tags") or []))

        fm["hardware_features"] = new_hw
        fm["techniques"] = new_tech
        fm["kernel_types"] = new_kt
        fm["architectures"] = new_arch
        fm["tags"] = new_tags
        if dirty:
            fm["facet_source"] = "inferred"

        for key, cur in (("hw", "hardware_features"), ("tech", "techniques"),
                         ("kt", "kernel_types")):
            if fm.get(cur):
                after[key] += 1

        if dirty and not args.dry_run:
            new_yaml = yaml.dump(fm, sort_keys=False, allow_unicode=True,
                                 default_flow_style=False)
            new_text = f"---\n{new_yaml}---\n" + text.split("---", 2)[2].lstrip("\n")
            # safer: rebuild using regex boundaries
            m = re.match(r'^---\s*\r?\n.*?\r?\n---\s*\r?\n(.*)', text, re.DOTALL)
            bodypart = m.group(1) if m else ""
            md.write_text(f"---\n{new_yaml}---\n\n{bodypart}", encoding="utf-8")
        if dirty:
            changed += 1

    n = len(pages)
    def pct(x): return f"{100*x//n}%" if n else "0%"
    print(f"PR pages: {n}")
    print(f"  hardware_features non-empty: {before['hw']} ({pct(before['hw'])}) "
          f"-> {after['hw']} ({pct(after['hw'])})")
    print(f"  techniques non-empty:        {before['tech']} ({pct(before['tech'])}) "
          f"-> {after['tech']} ({pct(after['tech'])})")
    print(f"  kernel_types non-empty:      {before['kt']} ({pct(before['kt'])}) "
          f"-> {after['kt']} ({pct(after['kt'])})")
    print(f"  pages changed: {changed}" + ("  (dry-run, nothing written)" if args.dry_run else ""))


if __name__ == "__main__":
    main()
