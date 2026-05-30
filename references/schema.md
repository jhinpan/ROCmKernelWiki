# Schema Reference

Condensed reference for ROCmKernelWiki's controlled vocabulary and page schemas.
Full definitions live in [`../data/schemas.yaml`](../data/schemas.yaml) and
[`../data/tags.yaml`](../data/tags.yaml).

## Page Types and IDs

Every page has a unique `id` with a type-specific prefix:

| Type | ID Prefix | Path | Purpose |
|------|-----------|------|---------|
| source-pr | `pr-<repo>-<N>` | `sources/prs/<repo>/PR-<N>.md` | A merged PR from a tracked ROCm repo |
| source-doc | `doc-*` | `sources/docs/` | Official AMD/ROCm docs, ISA guides, papers |
| source-blog | `blog-*` | `sources/blogs/` | ROCm/community blog posts, tutorials |
| source-ref | `ref-*` | `sources/refs/` | Reference repositories (FlyDSL, gcnasm, CK, …) |
| wiki-hardware | `hw-*` | `wiki/hardware/` | CDNA/RDNA hardware feature pages |
| wiki-technique | `technique-*` | `wiki/techniques/` | Optimization techniques |
| wiki-kernel | `kernel-*` | `wiki/kernels/` | Kernel case studies with perf claims |
| wiki-pattern | `pattern-*` | `wiki/patterns/` | Problem → solution diagnosis |
| wiki-language | `lang-*` | `wiki/languages/` | DSL / language guides |
| wiki-migration | `migration-*` | `wiki/migration/` | CUDA→ROCm and gfx→gfx migration |

## Required Frontmatter by Type

### source-pr
```yaml
id: pr-composable_kernel-1234
repo: ROCm/composable_kernel
pr: 1234
title: "Add gfx950 FP8 GEMM pipeline"
author: username
date: 2026-03-12
url: https://github.com/ROCm/composable_kernel/pull/1234
source_category: upstream-code
architectures: [gfx950]
tags: [fp8, mfma]
techniques: []            # may be empty
hardware_features: [mfma, fp8]
kernel_types: [fp8-gemm]
languages: [composable-kernel, hip]
captured_at: 2026-05-15
status: merged
merge_sha: abc123def456
inclusion_reason: "kernel path '...'; keyword 'fp8'"
changed_paths: [...]
```

### wiki-kernel (must have `performance_claims`)
```yaml
id: kernel-fp8-gemm
title: "FP8 Block-Scaled GEMM on CDNA4"
type: kernel
architectures: [gfx950]
tags: [fp8-gemm, fp8, mfma, block-scale]
confidence: source-reported
reproducibility: snippet
kernel_types: [fp8-gemm, gemm]
languages: [composable-kernel, hip]
related: [hw-mfma, hw-mxfp, technique-mfma-pipelining]
sources: [blog-fp8-gemm-cdna4, doc-cdna4-whitepaper, hw-mxfp]
performance_claims:
  - gpu: MI355X
    dtype: fp8
    shape: "M=N=K=8192"
    metric: TFLOPS
    value: 4200
    source_id: blog-fp8-gemm-cdna4
```

### wiki-pattern (diagnostic flow)
```yaml
id: pattern-bank-conflicts
title: "LDS Bank Conflicts"
type: pattern
tags: [lds, bank-conflict-avoidance]
symptoms: [bank-conflicts, lds-bound]
candidate_techniques: [technique-lds-swizzling, technique-bank-conflict-avoidance]
related: [hw-lds]
sources: [hw-lds, doc-cdna3-isa]
```

## Confidence Levels

- **`verified`**: Requires `evidence_basis` with ≥1 `official-doc` + ≥1 `upstream-code`/`paper`. Enforced by validator.
- **`source-reported`**: Cited by ≥1 authoritative source (AMD doc, major blog, major repo).
- **`inferred`**: Synthesized from multiple sources, no single authoritative one.
- **`experimental`**: Undocumented / version-sensitive. Note ROCm version.

## Reproducibility Levels

For `wiki-technique`, `wiki-kernel`, `wiki-language`, must be ≥ `snippet`.

| Level | Meaning |
|-------|---------|
| `concept` | Text only |
| `pseudocode` | Language-agnostic algorithm |
| `snippet` | Compilable code fragment (validator checks a fenced code block exists) |
| `runnable` | Self-contained buildable example |
| `benchmarked` | Runnable + perf numbers with env metadata |

## Controlled Vocabulary (excerpt)

All values in these frontmatter fields must appear in
[`../data/tags.yaml`](../data/tags.yaml):

- **architectures**: gfx942, gfx950, gfx1201, gfx90a, gfx1100, gfx1250
- **hardware_features**: matrix-core, mfma, wmma, lds, ds-instructions, buffer-instructions, global-instructions, async-copy, s-waitcnt, sgpr, vgpr, agpr, wave64, wave32, dpp, swizzle, permute, fp8, fp6, fp4, mxfp, bf16, fp16, int8, block-scale, xcd, l2-cache, infinity-cache, hbm3, …
- **techniques**: lds-double-buffering, mfma-pipelining, lds-swizzling, bank-conflict-avoidance, vectorized-loads, buffer-oob-guard, wave-reduce, occupancy-tuning, vgpr-budgeting, split-k, stream-k, preshuffle-layout, fine-grained-quantization, kernel-fusion, persistent-kernel, …
- **kernel_types**: gemm, hgemm, fp8-gemm, grouped-gemm, gemv, attention, flash-attention, paged-attention, mla, moe, fused-moe, rmsnorm, softmax, rope, kv-cache, all-reduce, transpose, bandwidth-bench, …
- **languages**: hip, cpp, gcn-asm, rocdl, composable-kernel, flydsl, mlir, triton, python
- **source_category**: official-doc, upstream-code, paper, benchmark-blog, reference-repo, community-note

## Canonical Aliases (from data/aliases.yaml)

When the user asks about:
- MI300 / MI300X / CDNA3 → architecture `gfx942`
- MI350 / MI355X / CDNA4 → architecture `gfx950`
- R9700 / RDNA4 → architecture `gfx1201`
- XDLOP / "matrix cores" → `mfma`
- "shared memory" / groupshared → `lds`
- waitcnt / vmcnt / lgkmcnt → `s-waitcnt`
- CK / ck-tile → `composable-kernel`
- microscaling / mxfp4 / mxfp8 → `mxfp`
