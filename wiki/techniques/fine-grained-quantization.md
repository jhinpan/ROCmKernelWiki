---
id: technique-fine-grained-quantization
title: Fine-Grained FP8 Quantization & Block Scaling
type: technique
architectures:
- gfx942
- gfx950
tags:
- fine-grained-quantization
- block-scaling
- fp8
- mxfp
- block-scale
- fp8-gemm
- quantization
confidence: source-reported
reproducibility: snippet
hardware_features:
- fp8
- mxfp
- block-scale
- mfma
kernel_types:
- fp8-gemm
- quantization
languages:
- hip
- triton
related:
- hw-mxfp
- hw-mfma
- kernel-fp8-gemm
- technique-preshuffle-layout
- migration-gfx942-to-gfx950
sources:
- hw-mxfp
- kernel-fp8-gemm
- blog-fp8-gemm-cdna4
- doc-cdna4-isa
- doc-cdna4-whitepaper
- blog-amd-matrix-cores
implemented_by:
- pr-FlyDSL-554
- pr-composable_kernel-3603
- pr-aiter-580
- pr-aiter-2136
- pr-FlyDSL-153
- pr-FlyDSL-278
- pr-sglang-22338
---
# Fine-Grained FP8 Quantization & Block Scaling

## Overview

FP8 GEMM only reaches its 2× speedup over FP16 (see
[MFMA throughput](../hardware/mfma.md)) if the quantization error stays small
enough to keep the network's accuracy. The problem is dynamic range: an E4M3
value covers roughly ±448, but LLM activations and weights routinely contain
outliers that, under a single per-tensor scale, force the rest of the tensor
into the denormal/zero region and destroy accuracy.

**Fine-grained quantization** fixes this by giving *small groups of elements
their own scale factor* so each group can use the full FP8 mantissa. The
granularity is a spectrum:

| Granularity | Scale shape (for `A[M,K]`) | Cost | Typical use |
|---|---|---|---|
| per-tensor | `1` | cheapest, worst accuracy | static activation calib |
| per-token (per-row) | `[M]` | one reduction per row | dynamic activations |
| per-channel (per-col) | `[K]` (or `[N]` on weights) | weights, offline | static weights |
| 1×128 / 128×128 block | `[M, K/128]` / `[M/128, K/128]` | DeepSeek-style | activations + weights |
| MX block (32-wide) | `[M, K/32]`, **E8M0** scale | hardware-native gfx950 | MXFP8/6/4 |

The first four are *software* scaling: you dequantize/rescale in the GEMM
epilogue or main loop. The last one, **MX (microscaling)**, is decoded by the
matrix core itself on CDNA4 (gfx950) via the `v_mfma_scale_*` instructions — see
[MXFP](../hardware/mxfp.md).

## Per-token / per-channel dynamic quantization

The canonical recipe for `D = A·B` with FP8 inputs:

1. Quantize activations `A` **per token** (one scale per row, computed at
   runtime from the row's `amax`).
2. Quantize weights `B` **per channel** (one scale per output column, computed
   offline).
3. Run the FP8 MFMA GEMM to get an FP32 partial.
4. Rescale in the epilogue: `D[m,n] = acc[m,n] * a_scale[m] * b_scale[n]`.

Because the two scale vectors are *separable* (a function of `m` only and `n`
only), the rescale is a cheap rank-1 outer product applied once per output tile
— it does **not** enter the K-loop.

```cpp
// Per-token (per-row) dynamic FP8 quantization of an [M, K] activation tensor.
// One block per row; wave64, CDNA. Produces row-major FP8 + an FP32 scale/row.
// gfx942 FP8 is FNUZ; gfx950 FP8 is OCP — pick the matching __hip_fp8 type.
#include <hip/hip_runtime.h>
#include <hip/hip_fp8.h>

constexpr float FP8_E4M3_MAX = 448.0f;     // OCP E4M3 max magnitude

__global__ void quantize_per_token_fp8(const float* __restrict__ in,
                                       __hip_fp8_e4m3* __restrict__ out,
                                       float* __restrict__ row_scale,
                                       int K) {
    const int row = blockIdx.x;
    const int lane = threadIdx.x;                 // 0..255 (4 waves)
    const float* src = in + (size_t)row * K;

    // 1) per-row amax reduction
    float amax = 0.0f;
    for (int k = lane; k < K; k += blockDim.x)
        amax = fmaxf(amax, fabsf(src[k]));

    __shared__ float red[256];
    red[lane] = amax;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (lane < s) red[lane] = fmaxf(red[lane], red[lane + s]);
        __syncthreads();
    }
    const float scale = red[0] / FP8_E4M3_MAX;    // dequant multiplier
    const float inv   = scale > 0.0f ? 1.0f / scale : 0.0f;
    if (lane == 0) row_scale[row] = scale;
    __syncthreads();

    // 2) scale + cast to FP8
    __hip_fp8_e4m3* dst = out + (size_t)row * K;
    for (int k = lane; k < K; k += blockDim.x)
        dst[k] = __hip_fp8_e4m3(src[k] * inv);    // store quantized value
}
```

The dequant rule is symmetric: stored FP8 value `q` reconstructs as
`q * scale`, so the epilogue multiplies the FP32 accumulator by
`a_scale[m] * b_scale[n]`.

> **Encoding mismatch (gfx942 vs gfx950).** CDNA3 FP8 is **FNUZ** (no inf, single
> NaN, different bias); CDNA4 FP8 is **OCP** E4M3/E5M2. The `FP8_E4M3_MAX`
> constant and the quantized bit-patterns differ between the two — never reuse a
> serialized FP8 checkpoint across architectures without requantizing. See the
> [gfx942 → gfx950 migration notes](../migration/gfx942-to-gfx950.md).

## Block scaling (1×128 / 128×128)

Per-token/per-channel still leaves a whole row sharing one scale. DeepSeek-V3
style **block scaling** tightens this to a 1×128 tile for activations and a
128×128 tile for weights. The scale now varies *along K*, so it must be applied
**inside** the K-loop: accumulate FP8 MFMA over a 128-deep K-chunk into FP32,
multiply that partial by the chunk's scale, then add into the running FP32
accumulator.

```cpp
// Sketch of the scaled K-loop (one output tile). acc stays in FP32 AGPRs.
float acc = 0.0f;
for (int kb = 0; kb < K / 128; ++kb) {
    float chunk = 0.0f;                       // FP8 MFMA over 128 K-elements
    chunk = mfma_fp8_kchunk(A_q + kb, B_q + kb, chunk);
    acc += chunk * a_scale[m_blk][kb] * b_scale[n_blk][kb];   // rescale per block
}
```

This keeps the matrix core in pure FP8 for the inner product while letting the
effective scale track K-direction outliers — the accuracy win that makes FP8
training/inference viable for large models.

## MX block scaling on CDNA4 (hardware-native)

On gfx950 the 32-element MX block scale is consumed directly by the matrix
core, so there is **no software rescale in the loop** at all. You supply the
quantized `f8f6f4` operands plus an **E8M0** (power-of-two, 8-bit exponent,
bias 127) scale per 32-wide block, and `ABID[0]=1` enables scaling:

```ptx
; gfx950: per-32-block scaled FP8/FP6/FP4 MMA, FP32 accumulate.
; A/B element formats are selected via CBSZ/BLGP (E4M3/E5M2/E2M3/E3M2/E2M1);
; one E8M0 scale operand applies per MX block.
v_mfma_scale_f32_16x16x128_f8f6f4 a[...], b[...], c[...]   ; ABID[0]=1
```

Because E8M0 is a pure exponent, MX rescaling is exact (a power-of-two shift)
and free in hardware. Use the `v_cvt_scalef32_pk_*` conversion ops (optionally
`_sr_` for stochastic rounding) to produce the packed narrow operands. Full
encoding details live on the [MXFP hardware page](../hardware/mxfp.md); a
worked end-to-end kernel is in [FP8 block-scaled GEMM](../kernels/fp8-gemm.md).

## Choosing a granularity

- **Inference, weights static, activations dynamic** → per-channel weights +
  per-token activations. Cheapest that stays accurate for most models.
- **Aggressive (sub-8-bit or hard models)** → 1×128 / 128×128 block scaling, or
  native MXFP on gfx950.
- **gfx950 available** → prefer MX block scale: the rescale is in silicon and
  unlocks the MXFP6/MXFP4 throughput tiers (up to 10 PF dense on MI355X).

Watch the cost of the quant pass itself: a separate quantize kernel reads and
rewrites the whole tensor in HBM. For decode-shaped GEMMs this can dominate, so
fuse quantization into the producer (the previous layer's epilogue) when
possible — see [kernel fusion](kernel-fusion.md).

## Sources

- [MXFP / block-scaled FP8-FP6-FP4 (gfx950)](../hardware/mxfp.md)
- [FP8 block-scaled GEMM kernel](../kernels/fp8-gemm.md)
- [FP8 GEMM on CDNA4 (ROCm blog)](https://rocm.blogs.amd.com/)
- [AMD CDNA4 ISA Reference Guide](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-cdna4-instruction-set-architecture.pdf)
- [AMD CDNA4 Architecture Whitepaper](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-4-architecture-whitepaper.pdf)
- [AMD Matrix Cores (ROCm blog)](https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores/README.html)
