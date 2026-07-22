---
id: kernel-fused-moe
title: Fused MoE — Gate-Up + SiLU + Down with FP8 (CDNA)
type: kernel
architectures:
- gfx942
- gfx950
tags:
- fused-moe
- moe
- grouped-gemm
- fp8
- kernel-fusion
- mfma
- block-scale
confidence: source-reported
reproducibility: runnable
artifact_dir: examples/fused-moe
kernel_types:
- fused-moe
- moe
- grouped-gemm
- fp8-gemm
languages:
- hip
- triton
related:
- kernel-grouped-gemm
- technique-kernel-fusion
- hw-mfma
- kernel-fp8-gemm
- technique-fine-grained-quantization
- lang-triton-amd
sources:
- ref-aiter
- blog-triton-amd
- blog-fp8-gemm-cdna4
- doc-mi300x-datasheet
- doc-cdna3-isa
performance_claims:
- gpu: MI300X
  dtype: fp8
  metric: speedup-vs-unfused
  value: ~1.3-1.6x end-to-end MoE layer vs separate gate/up/down GEMMs + activation
  source_id: ref-aiter
  confidence: source-reported
- gpu: MI300X
  dtype: fp8
  metric: peak-fp8-tensor
  value: 2615 TFLOPS dense FP8 ceiling (expert GEMMs are the dominant term)
  source_id: doc-mi300x-datasheet
  confidence: source-reported
- gpu: MI355X
  dtype: fp8
  metric: peak-fp8-tensor
  value: 5.0 PFLOPS dense OCP-FP8 ceiling (gfx950)
  source_id: blog-fp8-gemm-cdna4
  confidence: inferred
implemented_by:
- pr-sglang-24816
- pr-vllm-36286
- pr-vllm-36022
- pr-composable_kernel-2978
- pr-composable_kernel-3259
- pr-composable_kernel-2913
- pr-composable_kernel-2878
- pr-composable_kernel-2466
---
# Fused MoE — Gate-Up + SiLU + Down with FP8 (CDNA)

## Overview

A Mixture-of-Experts (MoE) FFN layer routes each token to a small number of
experts (`top_k` of `E`), then for each selected expert runs a two-GEMM gated
MLP:

```text
h      = SiLU(X · W_gate) * (X · W_up)      # "gate-up" projection + activation
y      = h · W_down                          # "down" projection
out    = sum over top_k experts (weighted by router probs)
```

On a GPU this is dominated by two **grouped GEMMs** (one per projection) where
the per-expert M dimension is data-dependent (it equals how many tokens the
router assigned to that expert). The naive implementation materializes large
intermediate tensors in HBM and launches separate kernels for gate, up, SiLU,
and down — burning bandwidth and launch latency. A **fused MoE** kernel keeps the
gate/up partial sums in registers/LDS, applies SiLU·mul inline, and feeds the
down-projection without a round-trip to HBM. This page covers the FP8 variant as
shipped in AMD's [AITER](../../sources/refs/ref-aiter.md) and the Triton AMD
backend.

This kernel is the MoE-specific composition of two general building blocks:
[grouped GEMM](grouped-gemm.md) (the expert-batched matmul + scheduling) and
[kernel fusion](../techniques/kernel-fusion.md) (folding the activation and,
where possible, the down-projection epilogue into the GEMM). The matmul math
itself runs on [MFMA matrix cores](../hardware/mfma.md).

## Why FP8 here

MoE layers are weight-heavy: total expert weights can dwarf attention. Storing
`W_gate/W_up/W_down` in FP8 halves weight bandwidth versus BF16 and doubles
matrix-core throughput (`v_mfma_f32_16x16x32_fp8_fp8` issues 16384 FLOPs vs 8192
for the FP16 `16x16x16` op — see [MFMA](../hardware/mfma.md)). Two caveats that
the kernel must respect:

- **gfx942 FP8 is FNUZ**, gfx950 FP8 is **OCP** — the encodings are not
  bit-compatible. Quantized expert weights are architecture-specific.
- Accuracy needs **fine-grained (per-block) scaling** of activations and
  weights; a single per-tensor scale loses too much for MoE routing. See
  [fine-grained quantization](../techniques/fine-grained-quantization.md). The
  GEMM accumulates in FP32 and applies the row/column block scales in the
  epilogue before SiLU.

## Routing and the grouped layout

The router produces, per token, `top_k` `(expert_id, weight)` pairs. Before the
GEMMs, a sort/scatter step groups token rows by destination expert and builds an
`expert_offsets` prefix-sum so each expert's slice is contiguous. The fused GEMM
is then a **grouped GEMM**: one persistent grid where each workgroup looks up
which expert (and which M-tile within that expert) it owns. Padding each expert's
token count up to the tile's `BLOCK_M` keeps tiles aligned and lets out-of-range
rows be handled by [buffer-load OOB guards](../techniques/buffer-oob-guard.md)
rather than divergent branches.

## Triton fused gate-up + SiLU kernel (AMD backend)

The gate and up projections share the same `X` tile and the same expert row, so
they are computed in one kernel: load the activation tile once, run two MFMA
accumulations against the stacked `W_gate_up` weight, then fuse SiLU·mul in the
epilogue. Only `h` (half the width) is written out.

```python
import triton
import triton.language as tl

@triton.jit
def fused_moe_gate_up_silu(
    X, Wgu, Out,                      # X:[T,K] fp8, Wgu:[E,2N,K] fp8, Out:[T,N]
    x_scale, w_scale,                 # block scales (fine-grained quant)
    sorted_token_ids, expert_ids, num_valid,
    T, N, K,
    stride_xt, stride_xk,
    stride_we, stride_wn, stride_wk,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # rows for this tile come from the router-sorted token list
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    tok = tl.load(sorted_token_ids + offs_m, mask=offs_m < num_valid, other=0)
    valid = offs_m < num_valid
    e = tl.load(expert_ids + pid_m)          # which expert this M-tile maps to

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    x_ptr = X + tok[:, None] * stride_xt + offs_k[None, :] * stride_xk
    # gate weights are rows [0:N), up weights rows [N:2N) of the stacked tensor
    g_ptr = Wgu + e * stride_we + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk
    u_ptr = g_ptr + N * stride_wn

    acc_g = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    acc_u = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        x = tl.load(x_ptr, mask=valid[:, None], other=0.0)
        g = tl.load(g_ptr)
        u = tl.load(u_ptr)
        acc_g += tl.dot(x, g.T)               # -> v_mfma_f32_16x16x32_fp8_fp8
        acc_u += tl.dot(x, u.T)
        x_ptr += BLOCK_K * stride_xk
        g_ptr += BLOCK_K * stride_wk
        u_ptr += BLOCK_K * stride_wk

    # apply per-block dequant scales, then fuse SiLU(gate) * up
    xs = tl.load(x_scale + tok, mask=valid, other=0.0)[:, None]
    gs = tl.load(w_scale + e * (2 * N) + offs_n)[None, :]
    us = tl.load(w_scale + e * (2 * N) + N + offs_n)[None, :]
    gate = acc_g * xs * gs
    up   = acc_u * xs * us
    h = (gate * tl.sigmoid(gate)) * up        # SiLU(gate) * up

    o_ptr = Out + tok[:, None] * N + offs_n[None, :]
    tl.store(o_ptr, h.to(Out.dtype.element_ty), mask=valid[:, None])
```

`tl.dot` lowers to `v_mfma_f32_16x16x32_fp8_fp8` on the AMD backend. Useful knobs
(see [Triton AMD](../languages/triton-amd.md)): `matrix_instr_nonkdim=16` to pin
the 16×16 MFMA, `waves_per_eu` to trade occupancy against
[VGPR pressure](../patterns/vgpr-pressure.md), and `num_stages` for the
software pipeline. On gfx950 the K-loop loads can use
[direct-to-LDS async copy](../hardware/async-copy-lds.md) to overlap weight
streaming with MFMA.

## The down projection

The down GEMM `y = h · W_down` consumes `h` produced above. Two strategies:

1. **Two-kernel fusion (common).** Gate-up+SiLU is one kernel; down is a second
   grouped GEMM that fuses the router-weight scaling and the top_k reduction
   (scatter-add back to the token's output row) into its epilogue. `h` lives in
   HBM but at half the gate-up width, and is freshly FP8-quantized for the down
   matmul. This is AITER's default split because the two GEMMs have different
   `(M,N,K)` and tile shapes.
2. **Single-kernel fusion.** Keep `h` in LDS and chain the down-projection in the
   same workgroup. This eliminates the `h` round-trip entirely but couples the
   two tile shapes and raises LDS/register pressure; it pays off mainly in the
   memory-bound **decode** regime (small M).

```cpp
// HIP epilogue sketch for the down projection: dequant -> route-weight -> scatter-add
// acc holds FP32 y-tile for one (expert, M-tile). topk_w[row] is the router prob.
__device__ void moe_down_epilogue(float* acc, const float* y_scale,
                                  const float* topk_w, const int* dst_row,
                                  __hip_fp8_e4m3_fnuz* /*unused*/, float* out,
                                  int m, int n, int N, bool valid) {
    if (!valid) return;                          // OOB row guard (padded tile)
    float y = acc[m * /*tileN*/ n + n] * y_scale[m];
    y *= topk_w[m];                              // weight this expert's contribution
    atomicAdd(&out[dst_row[m] * N + n], y);      // top_k reduction into token row
}
```

On gfx942 use the FNUZ FP8 types (`__hip_fp8_e4m3_fnuz`); on gfx950 use the OCP
`__hip_fp8_e4m3` types. The `atomicAdd` reduction is correct but contended for
large `top_k`; AITER instead pre-sorts so each output row is written by a single
workgroup where possible, avoiding atomics.

## Scheduling and the tail effect

Because per-expert token counts are imbalanced, a static grid leaves some CUs
idle while a few hot experts dominate — a classic [tail effect](../patterns/tail-effect.md).
Mitigations used in practice:

- **Persistent / flat work scheduling**: launch one workgroup per CU and pull
  padded `(expert, M-tile)` work items from a global counter
  ([grouped-GEMM scheduling](grouped-gemm.md)).
- **Tile-count padding** to `BLOCK_M` so partial tiles don't serialize.
- For decode (1 token × top_k), the problem is memory-bound; favor small tiles,
  high occupancy, and fusing both projections to cut HBM traffic.

## Performance notes

- The expert GEMMs set the ceiling: dense FP8 peaks at **2615 TFLOPS on MI300X**
  and **5.0 PFLOPS on MI355X** ([datasheet](../../sources/docs/doc-mi300x-datasheet.md),
  [CDNA4 FP8 blog](../../sources/blogs/blog-fp8-gemm-cdna4.md)). Real MoE layers
  run well below peak because per-expert M is small and routing/quant overhead is
  non-trivial.
- Fusing SiLU·mul and the down-epilogue removes two HBM passes over the
  intermediates; AITER reports roughly **1.3–1.6×** end-to-end improvement for an
  FP8 MoE layer versus unfused gate/up/down + standalone activation
  ([AITER](../../sources/refs/ref-aiter.md)). Treat as `source-reported`; exact
  gains depend on `E`, `top_k`, hidden size, and batch.

## Runnable example

A portable, self-checking fp32 reference of this dataflow lives in
[`examples/fused-moe/`](../../examples/fused-moe/). It fuses the per-token path —
router GEMV → top-k gating + softmax → gate-up GEMV + SiLU·mul (kept in LDS) →
down GEMV with router-weighted reduction — in a single HIP kernel (one block per
token) and verifies against a CPU reference. It uses **no MFMA/FP8**, so unlike
the production kernel above it is a compact scalar reference. It builds and runs
on gfx950.
Production MoE replaces the per-token GEMVs with grouped GEMM on matrix cores and
FP8 weights (see [grouped GEMM](grouped-gemm.md)).

```bash
cd examples/fused-moe && ./build.sh
```

Expected output (captured on MI355X / gfx950):

```
Fused MoE (fp32, portable HIP)
  dims: T=64 D=128 N=256 E=8 top_k=2
  kernel time: 108.413 us/iter (200 iters)
  max abs err: 5.215e-08
  max rel err: 1.508e-03
PASS
```

## See also

- [Grouped GEMM for MoE experts](grouped-gemm.md)
- [Kernel fusion](../techniques/kernel-fusion.md)
- [MFMA matrix cores](../hardware/mfma.md)
- [FP8 block-scaled GEMM (gfx950)](fp8-gemm.md)
- [Fine-grained quantization](../techniques/fine-grained-quantization.md)

## Sources

- [AITER — AMD AI operator library (FusedMoE)](https://github.com/ROCm/aiter)
- [Triton on AMD GPUs](https://rocm.blogs.amd.com/artificial-intelligence/triton/README.html)
- [FP8 GEMM on CDNA4](https://rocm.blogs.amd.com/software-tools-optimization/fp8-gemm-cdna4/README.html)
- [AMD Instinct MI300X Datasheet](https://www.amd.com/content/dam/amd/en/documents/instinct-business-docs/product-briefs/amd-instinct-mi300x-platform-data-sheet.pdf)
- [AMD CDNA3 ISA Reference Guide](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf)
