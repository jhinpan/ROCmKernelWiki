# AMD CDNA / RDNA Kernel Optimization Knowledge Base

> Comprehensive, agent-queryable knowledge base for GPU kernel optimization on
> AMD Instinct (CDNA3 gfx942 / CDNA4 gfx950) and Radeon (RDNA4 gfx1201).
> Optimized for LLM agent retrieval. See [CLAUDE.md](CLAUDE.md) for schema and
> conventions, and [SKILL.md](SKILL.md) for skill usage.
>
> Inspired by and modeled on MIT Han Lab's
> [KernelWiki](https://github.com/mit-han-lab/KernelWiki).
>
> **Validated on real MI350X silicon (gfx950, ROCm 7.2)** — see [VERIFICATION.md](VERIFICATION.md)
> and the first-party [FlyDSL kernel profiling sweep](sources/refs/ref-flydsl-kernel-profiling.md)
> ([dashboard](https://jhinpan.github.io/flydsl-kernel-profiling/)).

## Recommended Query Tools (for LLM agents)

```bash
python3 scripts/query.py "<natural language>" [--tag <t>] [--type <kernel|technique|hardware|...>]
python3 scripts/get_page.py <page-id-or-path> [--follow-sources]
python3 scripts/grep_wiki.py "<regex>" [--only wiki|sources]
```

See [references/examples.md](references/examples.md) for worked query patterns.

## Quick Navigation

| I want to... | Go to |
|---|---|
| Fix a performance problem | [queries/by-problem.md](queries/by-problem.md) |
| Learn a technique | [queries/by-technique.md](queries/by-technique.md) |
| Use a hardware feature | [queries/by-hardware-feature.md](queries/by-hardware-feature.md) |
| See what a repo contributed | [queries/by-repo.md](queries/by-repo.md) |
| Write a kernel type | [queries/by-kernel-type.md](queries/by-kernel-type.md) |
| Use a language/DSL | [queries/by-language.md](queries/by-language.md) |

## Hardware Features

- [hw-mfma](wiki/hardware/mfma.md) — AMD Matrix Core (`v_mfma_*`) instructions
- [hw-mxfp](wiki/hardware/mxfp.md) — MXFP / block-scaled FP8-FP6-FP4 (gfx950)
- [hw-lds](wiki/hardware/lds.md) — Local Data Share (banks, conflicts)
- [hw-memory-instructions](wiki/hardware/memory-instructions.md) — buffer / global / flat
- [hw-async-copy-lds](wiki/hardware/async-copy-lds.md) — direct-to-LDS async copy
- [hw-s-waitcnt](wiki/hardware/s-waitcnt.md) — vmcnt / lgkmcnt / expcnt gating
- [hw-wavefront](wiki/hardware/wavefront.md) — wave64, EXEC, SGPR/VGPR/AGPR
- [hw-cross-lane](wiki/hardware/cross-lane.md) — DPP, ds_permute, permlane16
- [hw-chiplet-xcd](wiki/hardware/chiplet-xcd.md) — XCD chiplets, Infinity Cache, NUMA
- [hw-wmma](wiki/hardware/wmma.md) — RDNA4 WMMA matrix instructions

## Optimization Techniques

- [technique-lds-double-buffering](wiki/techniques/lds-double-buffering.md)
- [technique-mfma-pipelining](wiki/techniques/mfma-pipelining.md)
- [technique-lds-swizzling](wiki/techniques/lds-swizzling.md)
- [technique-bank-conflict-avoidance](wiki/techniques/bank-conflict-avoidance.md)
- [technique-vectorized-loads](wiki/techniques/vectorized-loads.md)
- [technique-buffer-oob-guard](wiki/techniques/buffer-oob-guard.md)
- [technique-wave-reduce](wiki/techniques/wave-reduce.md)
- [technique-occupancy-tuning](wiki/techniques/occupancy-tuning.md)
- [technique-vgpr-budgeting](wiki/techniques/vgpr-budgeting.md)
- [technique-split-k](wiki/techniques/split-k.md)
- [technique-stream-k](wiki/techniques/stream-k.md)
- [technique-preshuffle-layout](wiki/techniques/preshuffle-layout.md)
- [technique-fine-grained-quantization](wiki/techniques/fine-grained-quantization.md)
- [technique-kernel-fusion](wiki/techniques/kernel-fusion.md)
- [technique-persistent-kernel](wiki/techniques/persistent-kernel.md)

## Kernel Case Studies

- [kernel-ck-hgemm](wiki/kernels/ck-hgemm.md) — FP16 GEMM via CK / MFMA
- [kernel-fp8-gemm](wiki/kernels/fp8-gemm.md) — FP8 block-scaled GEMM (gfx950)
- [kernel-flash-attention-ck](wiki/kernels/flash-attention-ck.md) — FlashAttention-2 (CK-tile)
- [kernel-flydsl-flash-attention](wiki/kernels/flydsl-flash-attention.md) — FlyDSL flash attention (gfx950 dual-wave)
- [kernel-paged-attention](wiki/kernels/paged-attention.md) — Paged attention decode
- [kernel-fused-moe](wiki/kernels/fused-moe.md) — Fused MoE
- [kernel-mla-decode](wiki/kernels/mla-decode.md) — MLA decode (DeepSeek)
- [kernel-rmsnorm](wiki/kernels/rmsnorm.md) — Fused RMSNorm
- [kernel-bandwidth-microbench](wiki/kernels/bandwidth-microbench.md) — Bandwidth microbenchmark
- [kernel-vector-add-asm](wiki/kernels/vector-add-asm.md) — Hand-asm persistent vector add
- [kernel-flydsl-preshuffle-gemm](wiki/kernels/flydsl-preshuffle-gemm.md) — FlyDSL preshuffle GEMM
- [kernel-grouped-gemm](wiki/kernels/grouped-gemm.md) — Grouped GEMM for MoE
- [kernel-transpose-lds](wiki/kernels/transpose-lds.md) — LDS-staged transpose

## Problem → Solution Patterns

- [pattern-bank-conflicts](wiki/patterns/bank-conflicts.md)
- [pattern-low-occupancy](wiki/patterns/low-occupancy.md)
- [pattern-vgpr-pressure](wiki/patterns/vgpr-pressure.md)
- [pattern-memory-bound](wiki/patterns/memory-bound.md)
- [pattern-mfma-underutilized](wiki/patterns/mfma-underutilized.md)
- [pattern-tail-effect](wiki/patterns/tail-effect.md)
- [pattern-xcd-locality](wiki/patterns/xcd-locality.md)

## Languages & DSLs

- [lang-hip](wiki/languages/hip.md) — HIP kernel programming
- [lang-gcn-asm](wiki/languages/gcn-asm.md) — GCN / CDNA assembly
- [lang-composable-kernel](wiki/languages/composable-kernel.md) — CK / ck_tile
- [lang-flydsl](wiki/languages/flydsl.md) — FlyDSL layout DSL
- [lang-triton-amd](wiki/languages/triton-amd.md) — Triton (AMD backend)
- [lang-rocwmma](wiki/languages/rocwmma.md) — rocWMMA fragment API

## Migration Guides

- [migration-cuda-to-hip](wiki/migration/cuda-to-hip.md) — CUDA → HIP / ROCm
- [migration-gfx942-to-gfx950](wiki/migration/gfx942-to-gfx950.md) — CDNA3 → CDNA4
- [migration-wmma-vs-mfma](wiki/migration/wmma-vs-mfma.md) — RDNA WMMA vs CDNA MFMA

## Source Repositories

| Repository | Focus |
|---|---|
| [ROCm/composable_kernel](queries/by-repo.md) | CK / CK-tile GEMM, attention, MoE |
| [ROCm/aiter](queries/by-repo.md) | AI Tensor Engine: MoE, attention, GEMM, quant |
| [ROCm/hipBLASLt](queries/by-repo.md) | GEMM with epilogue fusion, FP8 |
| [ROCm/Tensile](queries/by-repo.md) | Assembly GEMM kernel generator |
| [ROCm/rocBLAS](queries/by-repo.md) | BLAS on ROCm |
| [ROCm/flash-attention](queries/by-repo.md) | FlashAttention CK/Triton kernels |
| [ROCm/FlyDSL](queries/by-repo.md) | MLIR-native layout DSL |
| [ROCm/triton](queries/by-repo.md) | Triton AMD backend |
