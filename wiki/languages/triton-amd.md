---
id: lang-triton-amd
title: Triton on AMD — tl.dot → MFMA and the AMD backend knobs
type: language
version_sensitive:
- vs-triton-kpack-gfx950
architectures:
- gfx942
- gfx950
tags:
- triton
- mlir
- python
- mfma
- matrix-core
- async-copy
- buffer-instructions
- mfma-pipelining
- occupancy-tuning
confidence: source-reported
reproducibility: snippet
languages:
- triton
- mlir
- python
related:
- hw-mfma
- hw-async-copy-lds
- technique-mfma-pipelining
- technique-occupancy-tuning
- lang-composable-kernel
sources:
- blog-triton-amd
- blog-triton-optimizations
- blog-gluon-gemm
- doc-llvm-amdgpu
- blog-gemm-optimization
implemented_by:
- pr-triton-614
- pr-aiter-2441
- pr-FlyDSL-139
- pr-composable_kernel-1705
- pr-composable_kernel-1262
- pr-aiter-3072
- pr-FlyDSL-346
- pr-triton-879
---
# Triton on AMD — `tl.dot` → MFMA and the AMD backend knobs

## Overview

Triton is a Python DSL for GPU kernels that lowers tile-level tensor ops through
an MLIR pipeline to a target backend. The **AMD backend** (`triton.backends.amd`,
upstreamed and shipped in ROCm) targets CDNA3 (gfx942), CDNA4 (gfx950) and RDNA4
(gfx1201). The same Python you write for NVIDIA runs on AMD, but the
performance-critical decisions — which matrix instruction `tl.dot` becomes, how
the K-loop is pipelined, and how global loads stage through LDS — are controlled
by a small set of **AMD-specific knobs** passed via `triton.Config` /
`@triton.autotune` or environment variables.

On CDNA the key fact is that `tl.dot(a, b)` lowers to **`v_mfma_*`** matrix-core
instructions issued per wavefront (64 lanes) — see [MFMA](../hardware/mfma.md).
On RDNA4 it lowers to `v_wmma_*` instead. Everything below is about steering that
lowering.

```python
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_M": 256, "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_M": 8},
            num_warps=8, num_stages=2,
            # ---- AMD backend knobs live here ----
            kwargs={"matrix_instr_nonkdim": 16, "waves_per_eu": 2, "kpack": 2},
        ),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def gemm_kernel(A, B, C, M, N, K,
                stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
                BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                GROUP_M: tl.constexpr):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    # grouped/swizzled program ordering improves L2 (and XCD) reuse
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_n = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = A + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = B + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        acc += tl.dot(a, b)               # -> v_mfma_f32_16x16x16_f16 (per nonkdim)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c_ptrs = C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, acc.to(tl.float16))
```

## The AMD knob matrix

| Knob | Where | Effect | Notes |
|---|---|---|---|
| `matrix_instr_nonkdim` | Config kwarg | Picks the MFMA tile size (the M=N "non-K" dim): `16` → `16x16x*`, `32` → `32x32x*` | `16` favors small/skinny tiles & occupancy; `32` packs more FLOPs/issue. Maps to the shapes in [MFMA](../hardware/mfma.md). |
| `waves_per_eu` | Config kwarg | Occupancy hint: requested waves per execution unit (SIMD) | Higher → more latency hiding but fewer VGPRs/wave; trades against register pressure. See [occupancy tuning](../techniques/occupancy-tuning.md). |
| `kpack` | Config kwarg | Packs 2 K-slices per LDS read to widen `ds_read` and feed MFMA back-to-back | **gfx942 only; deprecated/removed on gfx950** — the wider-K MFMA shapes make it redundant. |
| `num_stages` | `triton.Config` | Software-pipeline depth of the K-loop (prefetch distance) | On AMD this drives the **stream / async-copy** pipeline, not NVIDIA-style `cp.async` groups. |
| `num_warps` | `triton.Config` | Wavefronts per program (block). On CDNA a "warp" = wave64 | Block tile = `num_warps` × MFMA tile; sets VGPR/LDS footprint. |

Environment overrides exist for experimentation, e.g.
`TRITON_HIP_USE_BLOCK_PINGPONG=1` (ping-pong scheduling, below) and buffer-ops
toggles; prefer baking the choice into autotune `configs` for reproducibility.

## How `tl.dot` chooses an MFMA

The backend selects an `v_mfma_*` instruction from the operand dtypes and
`matrix_instr_nonkdim`:

- FP16/BF16 inputs, `nonkdim=16` → `v_mfma_f32_16x16x16_f16` / `_bf16`.
- FP16/BF16 inputs, `nonkdim=32` → `v_mfma_f32_32x32x8_*`.
- FP8 inputs (FNUZ on gfx942, OCP on gfx950) → `v_mfma_f32_16x16x32_fp8_*` /
  `32x32x16`.
- On gfx950 the wider-K shapes (`16x16x32` f16, `32x32x16` f16, and the
  `f8f6f4` path) are available; this is why `kpack` is no longer needed there.

If the chosen tile does not divide `BLOCK_M/N/K`, Triton pads with masked loads,
so pick block sizes that are multiples of the MFMA shape to avoid wasted lanes.

## Async copy and buffer ops

Two AMD memory passes matter for GEMM/attention performance:

- **Direct-to-LDS async copy.** With sufficient `num_stages`, the backend emits
  `global_load_lds_*` / `buffer_load ... lds` so HBM→LDS streaming **bypasses
  VGPRs**, freeing registers and overlapping load with MFMA. This is AMD's
  analog of NVIDIA `cp.async`; gfx950 widens it to 12/16-byte loads. See
  [direct-to-LDS](../hardware/async-copy-lds.md). Gating uses `s_waitcnt vmcnt`,
  not an mbarrier object.
- **Buffer ops (MUBUF).** When pointer arithmetic provably fits a 128-bit
  resource descriptor (V#), the backend lowers `tl.load`/`tl.store` to
  `buffer_load/store`, getting **branchless OOB handling** (OOB reads return 0,
  OOB writes drop) instead of per-element predication. This often removes the
  masking overhead on the K-loop boundary.

## Ping-pong scheduling (gfx942)

`TRITON_HIP_USE_BLOCK_PINGPONG=1` enables a **ping-pong** schedule where two wave
groups in a block alternate phases: while one group issues MFMAs, the other
issues `ds_read`/global loads, so the matrix unit and the memory/VALU pipes stay
busy simultaneously. It is most effective on large FP16/FP8 GEMM tiles on CDNA3;
measure both ways, as it can hurt small problems. The Gluon-based GEMM work
generalizes this into explicit warp-level scheduling — see the
[Gluon GEMM blog](https://rocm.blogs.amd.com/).

## Tuning workflow

1. Start from an autotune sweep over `BLOCK_M/N/K`, `num_warps ∈ {4,8}`,
   `num_stages ∈ {1,2,3}`, `matrix_instr_nonkdim ∈ {16,32}`.
2. Inspect the generated assembly to confirm the expected `v_mfma_*` and that
   async copies appear: `AMDGCN_ENABLE_DUMP=1 python kernel.py` (dumps AMDGCN),
   or read the LLVM/`amdgcn` IR via the Triton cache.
3. Profile with `rocprofv3` / Omniperf: check MFMA-unit busy %, LDS bank
   conflicts, and VGPR/occupancy. If VGPR-bound, lower the tile or raise
   `waves_per_eu`; if LDS-bound, revisit `kpack` (gfx942) and swizzling.
4. Lock the winning `triton.Config` into source for reproducibility.

```bash
# Dump the AMDGCN that tl.dot lowered to, to verify the MFMA choice
AMDGCN_ENABLE_DUMP=1 MLIR_ENABLE_DUMP=1 python gemm.py 2>&1 | grep v_mfma
```

> **Portability note.** `matrix_instr_nonkdim`, `kpack`, `waves_per_eu`, and the
> ping-pong/buffer-ops toggles are **AMD-only**. CUDA-targeted Triton ignores
> them; guard target-specific configs with `triton.runtime.driver.active` /
> `torch.cuda.get_device_properties(...).gcnArchName` so one autotune table can
> serve both vendors. The portable knobs are `BLOCK_*`, `num_warps`,
> `num_stages`.

## See also

- [MFMA matrix-core instructions](../hardware/mfma.md)
- [Direct-to-LDS async copy](../hardware/async-copy-lds.md)
- [MFMA pipelining technique](../techniques/mfma-pipelining.md)
- [Composable Kernel DSL](composable-kernel.md)

## Sources

- [Triton on AMD GPUs (ROCm blog)](https://rocm.blogs.amd.com/)
- [Optimizing Triton kernels for AMD (ROCm blog)](https://rocm.blogs.amd.com/)
- [Gluon-based GEMM on AMD (ROCm blog)](https://rocm.blogs.amd.com/)
- [LLVM AMDGPU backend documentation](https://llvm.org/docs/AMDGPUUsage.html)
- [GEMM optimization on AMD (ROCm blog)](https://rocm.blogs.amd.com/)
