---
name: rocm-kernel-wiki
description: Search and apply ROCmKernelWiki when optimizing AMD Instinct kernels for MI300/gfx942 or MI350/MI355X/gfx950. Use for MFMA, LDS, direct-to-LDS, s_waitcnt, FP8/FP6/FP4/MXFP, wave reductions, GEMM/attention/MoE, CUDA-to-HIP migration, and CK/CK-Tile/AITER/hipBLASLt/FlyDSL/Triton/HIP/GCN implementations. Also use for merged-PR evidence from composable_kernel, AITER, Tensile, rocBLAS, flash-attention, FlyDSL, Triton, vLLM, or SGLang. Do not use for other GPU architectures, NVIDIA-only kernels, host framework behavior, or ROCm installation and driver troubleshooting.
---

# ROCmKernelWiki — AMD CDNA Kernel Optimization Wiki

> **Corpus dates:** the merged-PR harvest runs through **2026-05-30** (per
> `data/refresh-cutoff.yaml`); doc/blog pages carry individual retrieval dates.
> The nod-ai AMDGPU optimization guide is synchronized through commit
> `efa471ae` on **2026-07-20**. Re-run the relevant harvest or source-sync work
> before advancing either boundary.

Query a structured, cross-referenced knowledge base of AMD GPU kernel
optimization for CDNA3 (gfx942 / MI300) and CDNA4
(gfx950 / MI350-MI355X). The repository retains **7,454 merged-PR references**,
including quarantined raw material for future architectures; the active layer
contains 54 synthesis pages, 21 doc/blog summaries, and
10 reference-repository studies.

> Inspired by, and modeled on, MIT Han Lab's
> [KernelWiki](https://github.com/mit-han-lab/KernelWiki) (the Blackwell/Hopper
> kernel knowledge base) — see the citation in the README.

## How To Query

Treat the directory containing this `SKILL.md` as `<skill-root>`. Keep the
user's working directory unchanged and invoke the query tools by absolute path;
they resolve the corpus from their own location. Set `<python>` to
`<skill-root>/.venv/bin/python` on POSIX or
`<skill-root>/.venv/Scripts/python.exe` on Windows when that install-time venv
exists; otherwise use an available Python 3 with `requirements.txt` installed.
Resolve every relative result path against `<skill-root>`.

Start with search output and load only the relevant pages. Do not read the full
7,454-page PR corpus into context.

### Path 1: Unified search (preferred for natural language)

```bash
"<python>" "<skill-root>/scripts/query.py" "how to pipeline MFMA on MI300"
"<python>" "<skill-root>/scripts/query.py" --tag mfma --type kernel
"<python>" "<skill-root>/scripts/query.py" --repo composable_kernel --architecture gfx950 --limit 20
"<python>" "<skill-root>/scripts/query.py" --symptom bank-conflicts --compact
```

Filters: `--type`, `--tag`, `--repo`, `--language`, `--architecture`,
`--symptom`, `--confidence`, `--synthesis`, `--limit`, `--compact`, `--paths-only`.
Results are ranked IDF-weighted with priors that surface curated wiki pages and
runnable examples above raw PR noise, and each hit shows a matched-text snippet.
Add `--synthesis` to restrict to curated wiki pages (skip the 7,454 PR sources).
`--tag` and `--architecture` accept aliases — `--tag XDLOP` matches `mfma`,
`--tag cp.async` matches `async-copy`, `--architecture MI300` matches `gfx942`,
`--architecture MI355X` matches `gfx950`.
Use `--include-out-of-scope` only for explicit recovery research into retained
raw material; do not turn those results into supported architecture claims.

### Path 2: Fetch a specific page by id or path

```bash
"<python>" "<skill-root>/scripts/get_page.py" kernel-fp8-gemm
"<python>" "<skill-root>/scripts/get_page.py" pr-composable_kernel-1234
"<python>" "<skill-root>/scripts/get_page.py" kernel-fp8-gemm --follow-sources
"<python>" "<skill-root>/scripts/get_page.py" pr-composable_kernel-1234 --include-code --summary
"<python>" "<skill-root>/scripts/get_page.py" hw-mfma --body-only
```

Every wiki page now carries `implemented_by:` (the real PRs that built it) and
every linked PR carries `related:` back to the synthesis page — so you can hop
between "what it is" and "how it was actually implemented". Use `--include-code
--summary` to read a PR's `diff_summary.md` (files + key changed lines) instead of
the full diff.

### Path 3: Regex text search across wiki bodies and PR pages

```bash
"<python>" "<skill-root>/scripts/grep_wiki.py" "v_mfma_f32_16x16x16"
"<python>" "<skill-root>/scripts/grep_wiki.py" "global_load_lds" --only sources
"<python>" "<skill-root>/scripts/grep_wiki.py" "ds_bpermute|mov_dpp" --any
```

### Path 4: Pre-built cross-reference indices

Auto-generated under `queries/`:

- `queries/by-problem.md` — symptom → pattern page → candidate techniques
- `queries/by-technique.md` — techniques → every page that uses them
- `queries/by-hardware-feature.md` — mfma/lds/async-copy/mxfp/… → pages
- `queries/by-kernel-type.md` — gemm/attention/moe/… → pages
- `queries/by-language.md` — hip/gcn-asm/composable-kernel/flydsl/triton → pages
- `queries/by-repo.md` — all 7,454 PRs across the tracked ROCm repos

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
6. **Do not treat the wiki as an ISA-manual replacement.** For exact encodings,
   ABI fields, or unsupported instructions, follow the pinned primary source.
7. Refuse architecture-specific conclusions outside gfx942/gfx950 unless the
   user explicitly requests retained raw material via `--include-out-of-scope`.

## Knowledge Base Contents (PR cutoff 2026-05-30)

- **7,454 PR reference pages** across ROCm/composable_kernel, aiter, Tensile,
  rocBLAS, flash-attention, FlyDSL, triton, plus ROCm-filtered vLLM/SGLang
- **54 active wiki synthesis pages** plus 3 quarantined pages retained for recovery
- **21 doc/blog summaries** + **10 reference-repository studies**
- **9 candidate ledgers** classifying every scanned PR (include/defer/exclude)
- **6 auto-generated query indices**
- **959 real upstream PR diffs** in `artifacts/` + **12 gfx950-first example suites** in `examples/`
- **Validator** `scripts/validate.py` — schema, vocabulary, link-integrity (0 errors)

## Quality Guarantees

- Active hardware claims identify their evidence class and primary source.
- Every technique/kernel/language page carries a real code snippet.
- Every PR page has `inclusion_reason` and `status: merged`.
- `verified` pages carry `evidence_basis` (official-doc + upstream-code/paper).
- 0 dangling internal references (enforced by the validator).
