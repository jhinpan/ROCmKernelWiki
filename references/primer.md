# Topic Map / Primer

A compact map of the knowledge base. Use this as a fast lookup table when the
question is broad — each row points to the canonical page to open.

All page IDs resolve via `python3 scripts/get_page.py <id>`. Paths are relative
to the wiki root (the directory containing `data/`, `wiki/`, `sources/`).

---

## Architecture identity (read this first)

| Marketing | gfx | Arch | FP8 encoding | Matrix unit | Wave |
|---|---|---|---|---|---|
| MI300A/X, MI325X | `gfx942` | CDNA3 | **FNUZ** | MFMA | wave64 |
| MI350X, MI355X | `gfx950` | CDNA4 | **OCP** (+FP6/FP4/MX) | MFMA | wave64 |

> The single most important portability fact: **gfx942 FP8 (FNUZ) is not
> bit-compatible with gfx950 FP8 (OCP)**. See `migration-gfx942-to-gfx950`.
> Other architectures are retained only in raw sources and quarantined pages;
> they are not part of the active correctness contract.

---

## Hardware Features

| Feature | Page ID | Notes |
|---|---|---|
| MFMA matrix cores | `hw-mfma` | `v_mfma_*`; per-wavefront `D=A·B+C`; AGPR accumulators; gfx942 & gfx950 shapes |
| MXFP / block-scaled FP | `hw-mxfp` | gfx950 `f8f6f4` + E8M0 block scales (FP8/FP6/FP4) |
| LDS (Local Data Share) | `hw-lds` | 64 kB/32-bank (gfx942) → 160 kB/64-bank (gfx950); bank conflicts |
| Memory instructions | `hw-memory-instructions` | buffer (V# + OOB) vs global vs flat |
| Direct-to-LDS async copy | `hw-async-copy-lds` | `buffer_load…lds` / `global_load_lds`; AMD's cp.async analog |
| `s_waitcnt` counters | `hw-s-waitcnt` | vmcnt / lgkmcnt / expcnt; async gating |
| Wavefront & registers | `hw-wavefront` | wave64, EXEC, SGPR/VGPR/AGPR, occupancy |
| Cross-lane ops | `hw-cross-lane` | DPP, ds_swizzle, ds_permute/bpermute, permlane16 (gfx950) |
| Chiplet / XCD | `hw-chiplet-xcd` | XCDs, per-XCD L2, Infinity Cache, NUMA, partition modes |

---

## Optimization Techniques

| Technique | Page ID | When to use |
|---|---|---|
| LDS double-buffering | `technique-lds-double-buffering` | Overlap HBM load with MFMA compute |
| MFMA pipelining | `technique-mfma-pipelining` | Keep matrix cores busy; 4-wave interleave |
| LDS swizzling | `technique-lds-swizzling` | Remove bank conflicts on tile loads |
| Bank-conflict avoidance | `technique-bank-conflict-avoidance` | Padding + ds_read2 |
| Vectorized loads | `technique-vectorized-loads` | Saturate HBM with 128-bit / non-temporal |
| Buffer OOB guard | `technique-buffer-oob-guard` | Branchless tile-boundary handling |
| Wave reduce | `technique-wave-reduce` | DPP + ds_bpermute reductions |
| Occupancy tuning | `technique-occupancy-tuning` | `waves_per_eu`, VGPR/LDS limits |
| VGPR budgeting | `technique-vgpr-budgeting` | Accumulator tile vs occupancy tradeoff |
| Split-K | `technique-split-k` | Small M/N, large K |
| Stream-K | `technique-stream-k` | Load balance across CUs/XCDs |
| Preshuffle layout | `technique-preshuffle-layout` | Remove runtime weight swizzle |
| Fine-grained quantization | `technique-fine-grained-quantization` | Per-token/block FP8 + MX scale |
| Kernel fusion | `technique-kernel-fusion` | Epilogue/adjacent-op fusion |
| Persistent kernel | `technique-persistent-kernel` | Amortize launch; L2 reuse |

---

## Kernel Case Studies

| Kernel | Page ID | Stack |
|---|---|---|
| FP16 GEMM (CK) | `kernel-ck-hgemm` | Composable Kernel / MFMA |
| FP8 block-scaled GEMM | `kernel-fp8-gemm` | gfx950 f8f6f4 + MX |
| FlashAttention-2 (CK-tile) | `kernel-flash-attention-ck` | CK-tile FMHA |
| Paged attention | `kernel-paged-attention` | AITER / vLLM |
| Fused MoE | `kernel-fused-moe` | AITER |
| MLA decode | `kernel-mla-decode` | AITER / DeepSeek |
| RMSNorm | `kernel-rmsnorm` | fused norm |
| Bandwidth microbench | `kernel-bandwidth-microbench` | gcnasm |
| Vector add (asm) | `kernel-vector-add-asm` | gcnasm hand-asm |
| FlyDSL preshuffle GEMM | `kernel-flydsl-preshuffle-gemm` | FlyDSL |
| Grouped GEMM | `kernel-grouped-gemm` | CK / MoE |
| LDS transpose | `kernel-transpose-lds` | gcnasm |

---

## Problem → Pattern (Diagnosis)

| Symptom | Pattern page | Candidate techniques |
|---|---|---|
| LDS bank conflicts | `pattern-bank-conflicts` | swizzling, bank-conflict-avoidance |
| Low occupancy | `pattern-low-occupancy` | occupancy-tuning, vgpr-budgeting |
| VGPR/AGPR pressure | `pattern-vgpr-pressure` | vgpr-budgeting, occupancy-tuning |
| Memory-bandwidth bound | `pattern-memory-bound` | vectorized-loads, double-buffering, fusion |
| Matrix cores idle | `pattern-mfma-underutilized` | mfma-pipelining, double-buffering, preshuffle |
| Tail effect / imbalance | `pattern-tail-effect` | stream-k, split-k, persistent-kernel |
| Poor XCD/L2 locality | `pattern-xcd-locality` | stream-k, persistent-kernel |

---

## Languages / DSLs

| DSL | Page ID | Notes |
|---|---|---|
| HIP | `lang-hip` | `__global__`, `hipLaunchKernelGGL`, `__builtin_amdgcn_*` |
| GCN/CDNA assembly | `lang-gcn-asm` | `.s` kernels, inline asm, mnemonics |
| Composable Kernel | `lang-composable-kernel` | CK / ck_tile tile DSL |
| FlyDSL | `lang-flydsl` | MLIR-native layout DSL |
| Triton (AMD) | `lang-triton-amd` | `tl.dot`→MFMA; `matrix_instr_nonkdim`, `waves_per_eu` |

---

## Migration

| Page ID | Notes |
|---|---|
| `migration-cuda-to-hip` | CUDA→HIP; cp.async→direct-LDS; mbarrier→s_waitcnt; wgmma→mfma |
| `migration-gfx942-to-gfx950` | CDNA3→CDNA4; FNUZ→OCP FP8; LDS/permlane/f8f6f4 |

---

## Source Repositories (PR coverage)

Run `python3 scripts/query.py --repo <name>` or open
[`../queries/by-repo.md`](../queries/by-repo.md). Tracked repos:
ROCm/composable_kernel, ROCm/aiter, ROCm/hipBLASLt, ROCm/Tensile, ROCm/rocBLAS,
ROCm/flash-attention, ROCm/FlyDSL, ROCm/triton, plus ROCm-filtered
vllm-project/vllm and sgl-project/sglang.
