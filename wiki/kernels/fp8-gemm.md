---
id: kernel-fp8-gemm
title: "FP8 Block-Scaled GEMM on CDNA4 (gfx950)"
type: kernel
architectures:
- gfx950
- gfx942
tags:
- fp8-gemm
- gemm
- fp8
- mxfp
- block-scale
- mfma
- matrix-core
- fine-grained-quantization
confidence: source-reported
reproducibility: snippet
kernel_types:
- fp8-gemm
- gemm
languages:
- composable-kernel
- hip
- triton
related:
- hw-mxfp
- hw-mfma
- kernel-ck-hgemm
- technique-fine-grained-quantization
- technique-mfma-pipelining
- migration-gfx942-to-gfx950
sources:
- blog-fp8-gemm-cdna4
- blog-4wave-fp8-gemm
- hw-mxfp
- hw-mfma
- doc-cdna4-whitepaper
performance_claims:
- gpu: MI355X
  dtype: OCP-FP8 (E4M3)
  metric: peak dense TFLOPS
  value: "5000 (5.0 PF)"
  source_id: doc-cdna4-whitepaper
  confidence: source-reported
- gpu: MI355X
  dtype: MXFP6 / MXFP4
  metric: peak dense TFLOPS
  value: "10000 (10 PF)"
  source_id: doc-cdna4-whitepaper
  confidence: source-reported
- gpu: MI355X
  dtype: OCP-FP8 (E4M3)
  metric: achieved GEMM TFLOPS (large square M=N=K)
  value: "~3500-4200 (≈70-84% of peak)"
  source_id: blog-fp8-gemm-cdna4
  confidence: inferred
---

# FP8 Block-Scaled GEMM on CDNA4 (gfx950)

## Overview

FP8 GEMM is the workhorse of low-precision LLM inference and training: matrix
multiply `D = A · B` where `A` and `B` are 8-bit floats and the accumulator is
FP32. On CDNA4 (gfx950 / MI350X / MI355X), FP8 is **OCP-compliant** (E4M3 /
E5M2), unlike CDNA3's FNUZ encoding, and the matrix core gains a unified
`f8f6f4` path plus hardware **microscaling (MX)** with per-block **E8M0**
scales. This page shows how to build a block-scaled FP8 GEMM that feeds the new
`v_mfma_*_f8f6f4` instructions and reaches a large fraction of the MI355X's
5 PFLOP FP8 peak.

Two precision regimes coexist on gfx950 and this kernel covers both:

1. **Plain OCP-FP8** with coarse (per-tensor or per-row) scales applied in the
   epilogue — maps to `v_mfma_f32_16x16x128_f8f6f4` / `v_mfma_f32_32x32x64_f8f6f4`.
2. **MXFP8** (true block scaling) with one E8M0 scale per 32-element block,
   consumed *in hardware* by the `v_mfma_scale_*` variants. See the
   [MXFP page](../hardware/mxfp.md) for the format details.

## Why block scaling

A single FP8 E4M3 value has ~3 bits of mantissa and a narrow dynamic range. A
per-tensor scale wastes that range whenever a tensor has outliers. **Fine-grained
quantization** instead assigns a scale to each small block (here 32 contiguous
K-elements), so each block uses the full FP8 range around its own magnitude. On
CDNA4 the scale is an **E8M0** exponent (8-bit, bias 127, pure power-of-two), and
the matrix core multiplies it in during accumulation — no separate dequant pass.
This is the
[fine-grained quantization technique](../techniques/fine-grained-quantization.md)
realized in silicon.

## The matrix instruction

The dense (unscaled) and scaled CDNA4 FP8 MMA shapes are:

```ptx
; Unified FP8/FP6/FP4 dense MMA, FP32 accumulate
v_mfma_f32_16x16x128_f8f6f4  a[...], b[...], c[...]
v_mfma_f32_32x32x64_f8f6f4   a[...], b[...], c[...]

; Microscaling variants: take per-block E8M0 scale operands for A and B
v_mfma_scale_f32_16x16x128_f8f6f4 ...
v_mfma_scale_f32_32x32x64_f8f6f4  ...
```

The element format of each operand is selected by repurposed `CBSZ` (matrix A)
and `BLGP` (matrix B) fields: `000`=E4M3, `001`=E5M2, `010`=E2M3(FP6),
`011`=E3M2(BF6), `100`=E2M1(FP4). Mixed A/B formats are legal — e.g. FP8 weights
× FP6 activations. For the scaled forms, `ABID[0]=1` enables scaling; with it
clear, all scales are forced to 1.0. Note the K dimension: **128** for the
16×16 form (vs only 32 for the CDNA3 `v_mfma_f32_16x16x32_fp8_fp8`), so each MMA
consumes 4× more K per issue — fewer instructions, higher matrix-core duty cycle.

## HIP: scaled MMA via the LLVM intrinsic

The scaled builtin is exposed through `__builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4`
(LLVM `llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4`). A single wavefront tile:

```cpp
#include <hip/hip_runtime.h>

using fp8x32  = __attribute__((__vector_size__(32))) unsigned char; // 32 bytes of E4M3
using f32x4   = __attribute__((__vector_size__(16))) float;

// One 16x16x128 block-scaled FP8 MMA across a wave64.
// scaleA / scaleB are packed E8M0 exponents (one per 32-element K-block).
__device__ f32x4 mfma_bscale_16x16x128(fp8x32 a, fp8x32 b, f32x4 acc,
                                       int scaleA, int scaleB)
{
    // opsel/format args: 0,0 select E4M3 for both A and B (CBSZ/BLGP=000);
    // the two scale operands carry the per-block E8M0 exponents.
    return __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
        a, b, acc,
        /*cbsz=*/0, /*blgp=*/0,
        /*scaleA=*/scaleA, /*scaleB=*/scaleB);
}
```

In practice you do **not** hand-write the register layout. Use Composable
Kernel, hipBLASLt, or Triton, and reserve the intrinsic for custom epilogues or
fused dequant.

## Composable Kernel: the production path

The ROCm FP8 GEMM blog builds the fast kernel with `ck_tile`. The pipeline is the
classic GEMM mainloop, retuned for the wide-K FP8 MMA:

```cpp
// ck_tile GEMM mainloop sketch (block tile M=256, N=256, K=128 per stage)
// - load_tile: HBM -> LDS via direct-to-LDS async copy (frees VGPRs)
// - then v_mfma_*_f8f6f4 over the K-loop, FP32 accumulators in AGPRs
for (int k0 = 0; k0 < K; k0 += KPerBlock) {
    auto a_lds = load_tile(a_block_window);   // async global_load_lds
    auto b_lds = load_tile(b_block_window);
    block_sync_lds();                          // s_barrier
    // unified f8f6f4 MMA; scales applied per 32-K block (MXFP8) or in epilogue
    c_acc = tile_mma(a_lds, b_lds, c_acc);
}
// epilogue: apply row/col scales, optional bias + activation, cast to bf16/fp8
store_tile(c_out_window, cast<bf16>(c_acc * row_scale * col_scale));
```

Key tuning levers (all discussed in the CK FP8 blog and the 4-wave note):

- **Block tile**: 256×256×128 is a common sweet spot; the wider K (128) reduces
  mainloop instruction count vs CDNA3's K=32.
- **Direct-to-LDS** async copy (`global_load_lds_dwordx4`, widened to 16 B on
  gfx950) streams A/B straight to LDS, bypassing VGPRs — see
  [async copy](../hardware/async-copy-lds.md). Combined with the larger
  **160 kB LDS/CU** on CDNA4, this enables deeper double buffering.
- **MFMA pipelining / ping-pong**: overlap the matrix core with global→LDS
  traffic via [`s_waitcnt vmcnt`](../hardware/s-waitcnt.md) gating; see
  [MFMA pipelining](../techniques/mfma-pipelining.md).

## The 4-wave schedule

The [4-wave FP8 GEMM note](../sources/blogs/blog-4wave-fp8-gemm.md) describes a
scheduling refinement: run **4 waves per workgroup** and statically partition
them so that while two waves drive the matrix core on the current K-stage, the
other two issue the next stage's `global_load_lds` copies. Because the FP8 MMA is
so dense (K=128 per issue) the bottleneck shifts to keeping LDS fed; explicit
wave roles plus `__builtin_amdgcn_sched_barrier` to pin instruction order keep
the matrix core from stalling. This is a software realization of wave
specialization on top of the standard double-buffered mainloop.

## Triton path

Triton's AMD backend emits the `f8f6f4` MMA from `tl.dot` when operands are FP8
and the target is gfx950. The relevant knobs:

```python
import triton
import triton.language as tl

@triton.jit
def fp8_gemm(a_ptr, b_ptr, c_ptr, M, N, K, ...,
             BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0); pid_n = tl.program_id(1)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_blk)   # tl.float8e4nv on gfx950 (OCP E4M3)
        b = tl.load(b_blk)
        acc += tl.dot(a, b)  # lowers to v_mfma_f32_*_f8f6f4
    tl.store(c_blk, (acc * scale).to(tl.bfloat16))
```

Set `matrix_instr_nonkdim=16` (or 32) to pick the MFMA shape, raise
`num_stages` for deeper software pipelining, and tune `waves_per_eu` for
occupancy. `kpack` is deprecated on gfx950 because the K=128 MMA already packs a
full cache line. See the [Triton AMD guide](../languages/triton-amd.md).

## Performance

From the [CDNA4 whitepaper](../sources/docs/doc-cdna4-whitepaper.md) and the
[FP8 GEMM blog](../sources/blogs/blog-fp8-gemm-cdna4.md):

| GPU | dtype | Metric | Value |
|---|---|---|---|
| MI355X | OCP-FP8 (E4M3) | dense peak | 5.0 PFLOPS |
| MI355X | MXFP6 / MXFP4 | dense peak | 10 PFLOPS |
| MI300X | FP8 (FNUZ) | dense peak | 2615 TFLOPS |

Large square FP8 GEMMs (M=N=K in the multiples-of-thousands range) achieve a
high fraction of the 5 PF peak once the mainloop is LDS-fed and pipelined;
small/skinny shapes are memory- or launch-bound and benefit from
[split-K](../techniques/split-k.md) or [stream-K](../techniques/stream-k.md)
scheduling instead. MXFP4/MXFP6 double the peak again (10 PF) by halving the
bytes per element, at the cost of accuracy — choose the format per layer.

> **CDNA3 vs CDNA4.** On gfx942 FP8 is FNUZ and the MMA is K=32
> (`v_mfma_f32_16x16x32_fp8_fp8`, 2615 TFLOPS peak); there is **no** hardware MX
> scaling, so block scales must be dequantized in software. Porting weights
> across the two requires re-encoding FP8 — see the
> [gfx942 → gfx950 migration](../migration/gfx942-to-gfx950.md).

## See also

- [MXFP / block-scaled formats](../hardware/mxfp.md)
- [MFMA matrix instructions](../hardware/mfma.md)
- [FP16 GEMM via CK](ck-hgemm.md)
- [Fine-grained quantization](../techniques/fine-grained-quantization.md)

## Sources

- [FP8 GEMM on AMD CDNA4 (ROCm blog)](https://rocm.blogs.amd.com/artificial-intelligence/fp8-gemm/README.html)
- [A 4-wave FP8 GEMM schedule (ROCm blog)](https://rocm.blogs.amd.com/software-tools-optimization/4wave-fp8-gemm/README.html)
- [AMD CDNA4 Architecture Whitepaper](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-4-architecture-whitepaper.pdf)
- [Matrix Core Programming on CDNA3 and CDNA4](https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores-cdna/README.html)
- [OCP Microscaling (MX) Formats Specification v1.0](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)
