---
name: ROCmKernelWiki
description: Use when the user asks about optimizing AMD Instinct / Radeon GPU kernels — MI300 (gfx942/CDNA3), MI350/MI355X (gfx950/CDNA4), or RDNA4 (gfx1201) — MFMA/matrix-core programming, LDS bank conflicts, direct-to-LDS async copy, s_waitcnt pipelining, FP8/FP6/FP4/MXFP block scaling, wave64 reductions, GEMM/FlashAttention/fused-MoE/paged-attention kernels, or Composable-Kernel/CK-tile/FlyDSL/Triton-AMD/rocWMMA/HIP/GCN-assembly. Also for concrete merged-PR references from ROCm/composable_kernel, aiter, hipBLASLt, Tensile, rocBLAS, flash-attention, FlyDSL, and triton. Do NOT use for generic CUDA/NVIDIA-only questions, host-side framework integration, or ROCm install/driver troubleshooting.
argument-hint: "[natural-language-question] | [--tag mfma --type kernel] | [page-id]"
allowed-tools: "Bash Read Grep Glob"
---

# ROCmKernelWiki — AMD CDNA / RDNA Kernel Optimization Wiki

> **Knowledge cutoff: 2026-05-15.** All upstream PR data, doc summaries, and
> blog summaries reflect upstream state on or before this date (per
> `data/refresh-cutoff.yaml`). Re-run the harvest tooling to advance the cutoff.

Query a structured, cross-referenced knowledge base of AMD GPU kernel
optimization for CDNA3 (gfx942 / MI300), CDNA4 (gfx950 / MI350-MI355X), and
RDNA4 (gfx1201) — **7,400+ merged-PR references**, ~53 wiki synthesis pages,
20 doc/blog summaries, and 8 reference-repository studies.

> Inspired by, and modeled on, MIT Han Lab's
> [KernelWiki](https://github.com/mit-han-lab/KernelWiki) (the Blackwell/Hopper
> kernel knowledge base) — see the citation in the README.

## When To Use This Skill

Trigger this skill when the user asks about:

- **AMD matrix-core programming** — `v_mfma_*` shapes/dtypes, AGPR accumulators,
  the gfx950 `f8f6f4` unified low-precision path, MX (E8M0) block scaling
- **CDNA memory model** — LDS bank conflicts (32-bank gfx942 vs 64-bank gfx950),
  buffer vs global vs flat, OOB branchless guards, direct-to-LDS async copy,
  `s_waitcnt` (vmcnt/lgkmcnt) pipelining
- **Kernel implementations** — CK/hipBLASLt GEMM, FP8 block-scaled GEMM,
  FlashAttention-2 (CK-tile), paged attention, fused MoE, MLA decode, RMSNorm,
  bandwidth microbenchmarks
- **Performance problems** — bank conflicts, low occupancy, VGPR/AGPR pressure,
  memory-bound, idle matrix cores, tail effects, XCD/L2 locality
- **DSLs & languages** — HIP, GCN/CDNA assembly, Composable Kernel / CK-tile,
  FlyDSL, Triton (AMD backend), rocWMMA
- **Migration** — CUDA→HIP (cp.async→direct-LDS, mbarrier→s_waitcnt, wgmma→mfma),
  gfx942→gfx950 (FNUZ→OCP FP8!), RDNA WMMA vs CDNA MFMA
- **PR references** — "how did CK/AITER/hipBLASLt/Triton implement X on MI300/MI350?"

Do NOT use this skill for:

- Generic CUDA / NVIDIA-only kernel questions (use KernelWiki for Blackwell/Hopper)
- Host-side framework integration (model loading, request routing, scheduling)
- ROCm installation, driver, or environment troubleshooting

## How To Query

**First, cd into the skill directory** (the clone root — where this `SKILL.md`
lives), then run the tools. The scripts auto-resolve the wiki root from their own
location, so once you `cd` there no environment variable is needed:

```bash
cd "$(dirname "$(find ~/.claude/skills -name SKILL.md -path '*ROCmKernelWiki*' | head -1)")"
# ...or just: cd ~/.claude/skills/ROCmKernelWiki
```

If you cannot `cd` (e.g. running from another working directory), call the
scripts by absolute path — they still resolve the wiki root correctly:

```bash
python3 ~/.claude/skills/ROCmKernelWiki/scripts/query.py --tag mfma --type kernel
```

All example commands below assume you have `cd`'d into the skill directory.

### Path 1: Unified search (preferred for natural language)

```bash
python3 scripts/query.py "how to pipeline MFMA on MI300"
python3 scripts/query.py --tag mfma --type kernel
python3 scripts/query.py --repo composable_kernel --architecture gfx950 --limit 20
python3 scripts/query.py --symptom bank-conflicts --compact
```

Filters: `--type`, `--tag`, `--repo`, `--language`, `--architecture`,
`--symptom`, `--confidence`, `--synthesis`, `--limit`, `--compact`, `--paths-only`.
Results are ranked IDF-weighted with priors that surface curated wiki pages and
runnable examples above raw PR noise, and each hit shows a matched-text snippet.
Add `--synthesis` to restrict to curated wiki pages (skip the 7,400+ PR sources).
`--tag` and `--architecture` accept aliases — `--tag XDLOP` matches `mfma`,
`--tag cp.async` matches `async-copy`, `--architecture MI300` matches `gfx942`,
`--architecture MI355X` matches `gfx950`.

### Path 2: Fetch a specific page by id or path

```bash
python3 scripts/get_page.py kernel-fp8-gemm           # wiki page lists "Implementing PRs"
python3 scripts/get_page.py pr-composable_kernel-1234  # PR page lists "Synthesized in" wiki pages
python3 scripts/get_page.py kernel-fp8-gemm --follow-sources
python3 scripts/get_page.py pr-composable_kernel-1234 --include-code --summary  # compact diff
python3 scripts/get_page.py hw-mfma --body-only
```

Every wiki page now carries `implemented_by:` (the real PRs that built it) and
every linked PR carries `related:` back to the synthesis page — so you can hop
between "what it is" and "how it was actually implemented". Use `--include-code
--summary` to read a PR's `diff_summary.md` (files + key changed lines) instead of
the full diff.

### Path 3: Regex text search across wiki bodies and PR pages

```bash
python3 scripts/grep_wiki.py "v_mfma_f32_16x16x16"
python3 scripts/grep_wiki.py "global_load_lds" --only sources
python3 scripts/grep_wiki.py "ds_bpermute|mov_dpp" --any
```

### Path 4: Pre-built cross-reference indices

Auto-generated under `queries/`:

- `queries/by-problem.md` — symptom → pattern page → candidate techniques
- `queries/by-technique.md` — techniques → every page that uses them
- `queries/by-hardware-feature.md` — mfma/lds/async-copy/mxfp/… → pages
- `queries/by-kernel-type.md` — gemm/attention/moe/… → pages
- `queries/by-language.md` — hip/gcn-asm/composable-kernel/flydsl/triton → pages
- `queries/by-repo.md` — all 7,400+ PRs across the tracked ROCm repos

### Path 5: Primer, schema, examples

- `references/primer.md` — topic map; read first when the question is broad.
- `references/schema.md` — frontmatter schema, confidence/reproducibility ladders,
  controlled vocabulary, canonical aliases.
- `references/examples.md` — 7 worked query patterns.

## Output Pattern

When answering from this KB:

1. **State the architecture.** A fact true on gfx942 may differ on gfx950 — most
   importantly, **gfx942 FP8 is FNUZ and gfx950 FP8 is OCP** (not bit-compatible),
   and LDS is 64 kB/32-bank on gfx942 vs 160 kB/64-bank on gfx950.
2. **Cite specific pages** with paths and IDs (e.g. `wiki/hardware/mfma.md`,
   `hw-mfma`).
3. **Follow `sources:`** to trace claims to PRs/docs/blogs/refs.
4. **Respect confidence** — `verified` > `source-reported` > `inferred` >
   `experimental`.
5. **Report performance claims with all fields** — gpu, dtype, shape, metric,
   value, source_id.

## Knowledge Base Contents (cutoff 2026-05-15)

- **7,400+ PR reference pages** across ROCm/composable_kernel, aiter, hipBLASLt,
  Tensile, rocBLAS, flash-attention, FlyDSL, triton, plus ROCm-filtered vLLM/SGLang
- **~53 wiki synthesis pages** — hardware, techniques, kernels, patterns,
  languages, migration
- **20 doc/blog summaries** + **8 reference-repository studies**
- **9 candidate ledgers** classifying every scanned PR (include/defer/exclude)
- **6 auto-generated query indices**
- **959 real upstream PR diffs** in `artifacts/` + **12 runnable, hipcc-compiled kernel examples** in `examples/`
- **Validator** `scripts/validate.py` — schema, vocabulary, link-integrity (0 errors)

## Quality Guarantees

- Every hardware fact traces to an official AMD ISA doc / whitepaper.
- Every technique/kernel/language page carries a real code snippet.
- Every PR page has `inclusion_reason` and `status: merged`.
- `verified` pages carry `evidence_basis` (official-doc + upstream-code/paper).
- 0 dangling internal references (enforced by the validator).
