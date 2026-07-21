---
id: technique-kernel-fusion
title: Kernel Fusion — Epilogues and Adjacent-Op Merging
type: technique
architectures:
- gfx942
- gfx950
tags:
- kernel-fusion
- epilogue-fusion
- fp8
- rmsnorm
- fused-moe
- data-reuse
confidence: source-reported
reproducibility: snippet
hardware_features:
- mfma
- agpr
- vgpr
kernel_types:
- gemm
- fp8-gemm
- fused-moe
- rmsnorm
- elementwise
languages:
- hip
- hipblaslt-api
related:
- kernel-fused-moe
- kernel-rmsnorm
- technique-fine-grained-quantization
- technique-persistent-kernel
- kernel-fp8-gemm
sources:
- ref-hipblaslt
- blog-hipblaslt-tuning
- ref-aiter
- ref-composable-kernel
- doc-mi300x-datasheet
implemented_by:
- pr-composable_kernel-2978
- pr-composable_kernel-1789
- pr-composable_kernel-1791
- pr-composable_kernel-2594
- pr-composable_kernel-2551
- pr-composable_kernel-1862
- pr-composable_kernel-3259
- pr-composable_kernel-1591
---
# Kernel Fusion — Epilogues and Adjacent-Op Merging

## What and why

Kernel fusion collapses several logically distinct operations into a **single GPU
launch** so that intermediate tensors never round-trip through HBM. The dominant
cost in transformer inference epilogues is not arithmetic — it is the bandwidth
spent writing a GEMM result to HBM, reading it back to add a bias, reading it
again to apply an activation, and reading it a third time to requantize. Each of
those passes streams the full output tensor across the 5.3 TB/s
([MI300X](../../sources/docs/doc-mi300x-datasheet.md)) HBM link. Fusion keeps the
tile resident in VGPRs/AGPRs and pays the HBM write **once**.

Two flavors matter in practice:

1. **Epilogue fusion** — fold bias, activation (GELU/ReLU/SiLU), residual add,
   and output (re)quantization into the GEMM kernel's store path, while the
   accumulator tile is still in registers.
2. **Adjacent-op fusion** — merge producer/consumer elementwise + reduction
   chains (e.g. RMSNorm + quantize, or MoE gate/up matmul + SiLU·mul) so a tile
   is loaded once and several ops run on it before it is written back.

Both reduce launch overhead and, more importantly, **HBM traffic** — the binding
constraint for memory-bound LLM ops.

## Roofline intuition

For an `M×N` FP16 GEMM output, an unfused `bias→act→quant` chain to FP8 reads and
writes the `M×N` tensor multiple times:

| Stage | Unfused HBM bytes (per M×N element) | Fused |
|---|---|---|
| GEMM store (FP16) | 2 W | — (stays in regs) |
| Bias add (R+W FP16) | 2 R + 2 W | 0 |
| Activation (R+W FP16) | 2 R + 2 W | 0 |
| Quantize → FP8 (R FP16 + W FP8) | 2 R + 1 W | — |
| **Fused store (FP8)** | — | **1 W** |
| **Total** | **~11 bytes** | **~1 byte** |

The fused kernel does the bias/act/quant math on registers between the last MFMA
and the final `buffer_store`, so the only HBM write is the 1-byte FP8 output —
roughly an order-of-magnitude reduction in epilogue traffic.

## hipBLASLt epilogues

[hipBLASLt](../../sources/refs/ref-hipblaslt.md) exposes fused epilogues directly
through `hipblasLtMatmul`, which computes
`D = Act(α · op(A)·op(B) + β·C + bias)`. The epilogue is selected with a single
attribute on the matmul descriptor — no extra launch, no intermediate tensor.

```cpp
#include <hipblaslt/hipblaslt.h>

// Fuse: D = GELU(alpha * A*B + bias), with a separate bias vector.
hipblasLtMatmulDesc_t desc;
hipblasLtMatmulDescCreate(&desc, HIPBLAS_COMPUTE_32F, HIP_R_32F);

// 1) Pick the fused epilogue (bias + GELU activation).
hipblasLtEpilogue_t epi = HIPBLASLT_EPILOGUE_GELU_BIAS;
hipblasLtMatmulDescSetAttribute(
    desc, HIPBLASLT_MATMUL_DESC_EPILOGUE, &epi, sizeof(epi));

// 2) Point the epilogue at the bias vector (length N, broadcast over M).
hipblasLtMatmulDescSetAttribute(
    desc, HIPBLASLT_MATMUL_DESC_BIAS_POINTER, &d_bias, sizeof(d_bias));

// 3) Single launch does GEMM + bias + GELU; no intermediate D in HBM.
hipblasLtMatmul(handle, desc,
                &alpha, dA, layoutA, dB, layoutB,
                &beta,  dC, layoutC, dD, layoutD,
                &algo, workspace, wsBytes, stream);
```

Commonly available epilogue enums include `BIAS`, `RELU`, `GELU`, `GELU_BIAS`,
`RELU_BIAS`, and `*_AUX` variants that additionally write the pre-activation
tensor (needed to recompute gradients in training). For FP8 output, set the `D`
layout dtype to the FP8 type and supply an output scale; note that gfx942 FP8 is
**FNUZ** while gfx950 FP8 is **OCP**, so the scale/encoding must match the target
arch (see [fine-grained quantization](fine-grained-quantization.md)). Tuning the
backing TensileLite solution is covered in the
[hipBLASLt tuning blog](../../sources/blogs/blog-hipblaslt-tuning.md).

## Hand-written epilogue fusion (HIP)

When you own the kernel, do the epilogue while the accumulator tile is live in
registers. The pattern: finish the MFMA accumulation, apply bias/activation on
the per-lane accumulator fragment, then a single vectorized store.

```cpp
// Per-thread accumulator fragment after the K-loop of an MFMA GEMM tile.
// acc[] holds this lane's slice of the 16x16 (or 32x32) C tile in VGPR/AGPR.
__device__ __forceinline__
void fused_epilogue(float acc[4],            // live accumulator, no HBM round-trip
                    const float* __restrict__ bias, int n0,
                    float out_scale,          // for FP8 requant
                    __hip_fp8_storage_t* __restrict__ Dq) // FP8 output
{
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        float x = acc[i] + bias[n0 + i];      // (1) bias add  — in register
        x = 0.5f * x * (1.0f + erff(x * 0.70710678f)); // (2) exact GELU
        float q = x * out_scale;              // (3) scale for quant
        // (4) single store: only the FP8 byte hits HBM
        Dq[i] = __hip_cvt_float_to_fp8(q, __HIP_SATFINITE, __HIP_E4M3_FNUZ);
    }
}
```

Everything between the last MFMA and the store stays in registers, so the bias,
activation, and quantization add essentially zero HBM traffic — the dominant
cost remains the single FP8 write.

## Adjacent-op fusion: norm + quant, MoE gate/up

The same principle applies to producer/consumer chains that are *not* GEMM
epilogues:

- **RMSNorm + quantize.** The [RMSNorm kernel](../kernels/rmsnorm.md) computes
  `1/rms` with a [wave reduction](wave-reduce.md), then in the **same** pass
  multiplies by the weight and emits FP8 with its scale — the activation row is
  read once and written once, instead of norm→HBM→quant→HBM.
- **Fused MoE.** The [fused-MoE kernel](../kernels/fused-moe.md) fuses the
  gate and up projections, the `SiLU(gate) * up` elementwise combine, and the
  output quantization into the grouped-GEMM epilogue, so the large intermediate
  `[tokens × inter]` activation never materializes in HBM. AMD's
  [AITER](../../sources/refs/ref-aiter.md) ships these as default LLM ops.

In [Composable Kernel](../../sources/refs/ref-composable-kernel.md), the epilogue
is a *tile element-wise op* plugged into the pipeline's `store_tile` stage, so
fusion is expressed at the tile level rather than hand-rolled per kernel.

## When fusion helps — and when it does not

**Fuse when:**

- The chain is memory-bound and the intermediate is large (epilogues, norms,
  activation chains). Bandwidth saved scales with the eliminated passes.
- The producer output fits in registers/LDS at tile granularity (true for GEMM
  epilogues and row-wise norms).

**Be careful when:**

- Fusion inflates **VGPR/AGPR pressure** and drops occupancy — a heavy epilogue
  can spill and erase the bandwidth win. Check register usage and consider
  [VGPR budgeting](vgpr-budgeting.md) / [occupancy tuning](occupancy-tuning.md).
- The fused op needs a **global reduction** that crosses tiles (e.g. softmax
  normalization across the full row when tiles split the row) — that may force a
  split-pass or [Stream-K](stream-k.md)-style scheme instead.
- A fused-but-untuned monolith beats nothing; but a well-tuned library epilogue
  (hipBLASLt) usually beats a naive hand fusion. Profile before committing.

A practical rule: fuse the **cheap, bandwidth-dominated tail** onto the
**expensive, compute-dominated body**. The MFMA body hides the elementwise
epilogue latency for free, and you delete every HBM round-trip the epilogue would
have cost.

## See also

- [Fused MoE kernel](../kernels/fused-moe.md)
- [RMSNorm fused kernel](../kernels/rmsnorm.md)
- [Fine-grained quantization](fine-grained-quantization.md)
- [Persistent kernels](persistent-kernel.md)

## Sources

- [hipBLASLt (rocm) — fused matmul epilogues](https://github.com/ROCm/hipBLASLt)
- [hipBLASLt tuning guide](https://rocm.blogs.amd.com/)
- [AITER — AMD AI operator library](https://github.com/ROCm/aiter)
- [Composable Kernel / ck_tile](https://github.com/ROCm/composable_kernel)
- [AMD Instinct MI300X datasheet](https://www.amd.com/en/products/accelerators/instinct/mi300/mi300x.html)
