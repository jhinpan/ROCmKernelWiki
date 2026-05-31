---
id: kernel-rmsnorm
title: "Fused RMSNorm (+ residual / quant) on CDNA"
type: kernel
architectures:
- gfx942
- gfx950
tags:
- rmsnorm
- layernorm
- elementwise
- reduction
- wave-reduce
- kernel-fusion
- fp8
- memory-bound
confidence: source-reported
reproducibility: runnable
artifact_dir: examples/rmsnorm
kernel_types:
- rmsnorm
- elementwise
languages:
- hip
- triton
hardware_features:
- lds
- dpp
- permute
- wave64
- fp8
techniques:
- wave-reduce
- kernel-fusion
- vectorized-loads
- fine-grained-quantization
related:
- technique-wave-reduce
- technique-kernel-fusion
- hw-cross-lane
- hw-lds
- kernel-fused-moe
sources:
- ref-aiter
- blog-triton-amd
- doc-mi300x-datasheet
- doc-rocm-hip-hw
- blog-triton-optimizations
performance_claims:
- gpu: MI300X
  dtype: bf16
  metric: HBM bandwidth utilization
  value: "~80-90% of 5.3 TB/s (memory-bound, large hidden)"
  source_id: doc-mi300x-datasheet
  shape: "[8192, 8192]"
  utilization: memory-bound
- gpu: MI300X
  dtype: bf16
  metric: speedup vs unfused norm + residual + quant
  value: "~1.5-2x from fusing residual-add and FP8 quant into the norm"
  source_id: ref-aiter
  baseline: "three separate elementwise launches"
- gpu: MI355X
  dtype: bf16
  metric: HBM bandwidth utilization
  value: "~80-90% of up to 8 TB/s"
  source_id: doc-mi300x-datasheet
  utilization: memory-bound
---

# Fused RMSNorm (+ residual / quant) on CDNA

## Overview

Root-Mean-Square LayerNorm (RMSNorm) is the normalization used by Llama,
Mistral, DeepSeek and most modern LLMs. For a row (token) vector `x` of length
`H`:

```
rms      = sqrt( (1/H) * Σ x_i^2 + eps )
y_i      = (x_i / rms) * w_i
```

There is no mean-subtraction and no bias — only a single reduction
(`Σ x_i^2`) and an elementwise rescale. That makes RMSNorm almost purely
**memory-bound**: the arithmetic is trivial, so performance is governed by how
close you get to HBM peak (5.3 TB/s on MI300X, up to 8 TB/s on MI355X) and how
many bytes you can avoid moving. The two levers are:

1. **An efficient in-wave reduction** for `Σ x_i^2` — see
   [wave-reduce](../techniques/wave-reduce.md) and the
   [cross-lane primitives](../hardware/cross-lane.md).
2. **Fusion** of the surrounding pointwise ops (residual-add, FP8/INT8
   output quantization) so the data is read/written once instead of three
   times — see [kernel fusion](../techniques/kernel-fusion.md).

AITER ships exactly these fused variants (`rmsnorm2d_fwd`,
`rmsnorm2d_fwd_with_add`, `rmsnorm2d_fwd_with_dynamicquant`, etc.) with both
Triton and HIP/CK backends; see [ref-aiter](../../sources/refs/ref-aiter.md).

## Parallelization strategy

Map **one block (one or more wavefronts) to one row**. `H` for LLMs is
typically 4096–16384, so a row is processed in vectorized chunks across the
block's lanes, then a single block-wide reduction produces the scalar `rms`.

- Each lane loads a strided set of elements with **128-bit vectorized loads**
  (`float4` / 8×bf16) to saturate the memory pipe.
- Accumulate `Σ x_i^2` in **FP32** even for bf16/fp16 inputs — squaring fp16
  loses precision and can bias the norm.
- Reduce within a wavefront with DPP/`ds_bpermute` (no LDS), then across
  wavefronts of the block through one LDS round-trip.
- Broadcast `1/rms`, rescale, and (optionally) add the residual and quantize
  to FP8 in the same pass.

> On CDNA `warpSize == 64`. Do not hardcode 32; query `warpSize` or
> `__AMDGCN_WAVEFRONT_SIZE__`. CDNA is wave64-only.

## HIP implementation (fused residual + RMSNorm)

```cpp
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>

// One block per row. BLOCK lanes cooperate over a row of H elements.
// Fuses: y = rmsnorm(x + residual) * weight, and writes back x+residual
// into `residual_out` so the next layer reuses it (classic pre-norm fusion).
template <int BLOCK>
__global__ void rmsnorm_add_kernel(
    __hip_bfloat16*       __restrict__ y,            // [R, H]
    __hip_bfloat16*       __restrict__ residual_out, // [R, H]  (x + residual)
    const __hip_bfloat16* __restrict__ x,            // [R, H]
    const __hip_bfloat16* __restrict__ residual,     // [R, H]
    const __hip_bfloat16* __restrict__ weight,       // [H]
    float eps, int H)
{
    const int row  = blockIdx.x;
    const int lane = threadIdx.x;               // 0 .. BLOCK-1
    const __hip_bfloat16* xr = x        + (size_t)row * H;
    const __hip_bfloat16* rr = residual + (size_t)row * H;
    __hip_bfloat16*       sr = residual_out + (size_t)row * H;

    // 1) Σ (x+res)^2 in fp32, while caching the summed value back to HBM.
    float local = 0.f;
    for (int i = lane; i < H; i += BLOCK) {
        float v = __bfloat162float(xr[i]) + __bfloat162float(rr[i]);
        sr[i]   = __float2bfloat16(v);          // residual_out = x + residual
        local  += v * v;
    }

    // 2) Wavefront reduction via DPP butterfly (no LDS traffic).
    //    See technique-wave-reduce / hw-cross-lane.
    for (int off = 32; off > 0; off >>= 1)
        local += __shfl_down(local, off);       // __shfl_* -> ds_bpermute/DPP

    // 3) Cross-wavefront reduction through LDS (BLOCK/64 partial sums).
    __shared__ float partials[BLOCK / 64];
    if ((lane & 63) == 0) partials[lane >> 6] = local;
    __syncthreads();

    float total = 0.f;
    if (lane < BLOCK / 64) total = partials[lane];
    if (lane < 64) {                            // first wave finalizes
        for (int off = 32; off > 0; off >>= 1)
            total += __shfl_down(total, off);
        if (lane == 0) partials[0] = rsqrtf(total / H + eps);
    }
    __syncthreads();
    const float inv_rms = partials[0];

    // 4) Rescale + apply weight (re-reads residual_out from L2, not HBM).
    for (int i = lane; i < H; i += BLOCK) {
        float v = __bfloat162float(sr[i]) * inv_rms;
        y[i + (size_t)row * H] =
            __float2bfloat16(v * __bfloat162float(weight[i]));
    }
}
```

Launch with `BLOCK` a multiple of 64 (256 or 512 are good defaults), one block
per row:

```cpp
constexpr int BLOCK = 256;
rmsnorm_add_kernel<BLOCK><<<R, BLOCK>>>(y, res_out, x, res, w, eps, H);
```

Notes for the kernel engineer:

- `__shfl_down` lowers to `ds_bpermute_b32` / DPP on AMD; for a hand-tuned
  path use `__builtin_amdgcn_mov_dpp` for the intra-row steps and
  `__builtin_amdgcn_ds_bpermute` for the cross-row step — see
  [cross-lane](../hardware/cross-lane.md). `v_permlane16_*` is **gfx950-only**.
- The residual sum is written once and re-read from **L2**, not HBM, so the
  fused kernel touches HBM ~3× less than three separate launches
  (read x, read res, write y) — this is the bandwidth win behind the
  ~1.5–2× speedup.
- Prefer `float4`/`bf16x8` vectorized loads when `H` is a multiple of the
  vector width; fall back to scalar tails. See
  [vectorized loads](../techniques/vectorized-loads.md).

## Triton implementation (with dynamic FP8 quant)

The same structure in Triton lets the AMD backend pick the MFMA-free reduction
and emit `buffer`/`global` loads. A common AITER-style variant produces an FP8
output plus a per-row scale (dynamic per-token quantization):

```python
import triton
import triton.language as tl

@triton.jit
def rmsnorm_quant(x_ptr, w_ptr, y_ptr, scale_ptr,
                  H: tl.constexpr, eps: tl.constexpr,
                  BLOCK: tl.constexpr, FP8_MAX: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < H

    x = tl.load(x_ptr + row * H + cols, mask=mask, other=0.0).to(tl.float32)
    # Σ x^2 -> block reduction (Triton lowers to DPP/ds_bpermute on AMD)
    inv_rms = tl.rsqrt(tl.sum(x * x, axis=0) / H + eps)
    w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    y = x * inv_rms * w

    # dynamic per-row FP8 (E4M3) quant: scale = amax / FP8_MAX
    amax = tl.max(tl.abs(y), axis=0)
    scale = amax / FP8_MAX
    q = (y / scale).to(tl.float8e4nv)          # gfx950 OCP; gfx942 -> fnuz
    tl.store(y_ptr + row * H + cols, q, mask=mask)
    tl.store(scale_ptr + row, scale)
```

On the AMD Triton backend, tune `BLOCK` (rounded up to a power of two ≥ `H`),
`num_warps`, and `waves_per_eu`; let the compiler emit buffer ops for the
branchless masked loads. See [blog-triton-amd](../../sources/blogs/blog-triton-amd.md)
and [Triton kernel optimizations](../../sources/blogs/blog-triton-optimizations.md).

> **FP8 type matters.** gfx942 FP8 is **FNUZ** (`tl.float8e4b8`-style /
> `__hip_fp8_e4m3_fnuz`); gfx950 is **OCP** E4M3 (`float8e4nv`). A quant scale
> computed for one is not bit-compatible with the other — pick the type by
> target arch.

## Performance notes

- **Roofline.** With one fp16/bf16 read + one write, the kernel moves
  `~4·R·H` bytes. At MI300X's 5.3 TB/s, an `[8192, 8192]` bf16 RMSNorm has a
  ~few-hundred-µs HBM floor; a well-vectorized fused kernel lands within
  ~10–20% of that. Anything far below means you are not vectorizing loads or
  occupancy is too low.
- **Why fuse.** Unfused pipelines launch residual-add, RMSNorm, and quant as
  three memory-bound kernels, each paying full HBM round-trips and launch
  overhead. Fusing collapses them to one read + one write — the dominant cost
  on a bandwidth-bound op (see [kernel fusion](../techniques/kernel-fusion.md)).
- **Reduction cost.** For `H ≥ 4096` the reduction is a small fraction of
  runtime, but a naive all-LDS reduction with bank conflicts can still stall a
  wave. Keep the intra-wave step in registers (DPP/`ds_bpermute`) and use LDS
  only for the `BLOCK/64` cross-wave partials — see
  [wave-reduce](../techniques/wave-reduce.md) and [LDS](../hardware/lds.md).
- **Small `H` / decode.** When `H` is small or the batch is tiny (decode), the
  op becomes latency-bound; raise occupancy
  ([occupancy tuning](../techniques/occupancy-tuning.md)) or batch multiple
  rows per block to amortize launch and reduction overhead.

## Runnable example

A portable, self-checking pure-HIP version of this kernel lives in
[`examples/rmsnorm/`](../../examples/rmsnorm/). It implements the one-block-per-row
strategy above — FP32 `Σ x²` accumulation, intra-wave `__shfl_down` reduction,
cross-wave finalize through LDS, optional `gamma` — and is **wave-size agnostic**
(uses the `warpSize` builtin), so the same source runs on gfx1201 (wave32) and
CDNA (wave64). It validates fp32 and fp16-IO paths against a CPU reference.

```bash
cd examples/rmsnorm
hipcc --offload-arch=gfx1201 -O3 -std=c++17 rmsnorm.hip.cpp -o rmsnorm && ./rmsnorm
# (./build.sh gfx942 cross-compiles for CDNA on this box)
```

Expected output (AMD Radeon RX 9070 XT, gfx1201, ROCm 7.2.3):

```
Device: AMD Radeon RX 9070 XT  warpSize=32
---------------------------------------------------------------
rmsnorm fp32           [ 1024 x  4096] gamma=1  max|err|=2.384e-07  PASS  0.0707 ms  475 GB/s
rmsnorm fp32 no-gamma  [  512 x  8192] gamma=0  max|err|=2.384e-07  PASS  0.0533 ms  630 GB/s
rmsnorm fp32 odd-H     [  300 x  4097] gamma=1  max|err|=2.384e-07  PASS
rmsnorm fp16 IO        [ 1024 x  4096] gamma=1  max|err|=4.884e-04  PASS  0.0588 ms  285 GB/s
rmsnorm fp16 IO big    [  256 x 16384] gamma=1  max|err|=4.884e-04  PASS  0.0551 ms  305 GB/s
---------------------------------------------------------------
ALL TESTS PASSED
```

(The GB/s numbers use scalar loads — a correctness/portability demo, not a peak
bandwidth result; vectorize loads as described above for peak.)

## See also

- [Wave-level reductions](../techniques/wave-reduce.md)
- [Kernel fusion](../techniques/kernel-fusion.md)
- [Cross-lane primitives (DPP / bpermute / permlane16)](../hardware/cross-lane.md)
- [Fused MoE kernel](fused-moe.md) (same fusion + quant patterns)

## Sources

- [AITER — AMD AI operator library (norm/quant fused ops)](https://github.com/ROCm/aiter)
- [Triton on AMD GPUs](https://rocm.blogs.amd.com/artificial-intelligence/triton/README.html)
- [AMD Instinct MI300X datasheet (HBM bandwidth, peak figures)](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/data-sheets/amd-instinct-mi300x-data-sheet.pdf)
- [ROCm HIP programming guide (warpSize, builtins)](https://rocm.docs.amd.com/projects/HIP/en/latest/)
- [Triton kernel performance optimization on AMD](https://rocm.blogs.amd.com/artificial-intelligence/triton-kernel-optimization/README.html)
