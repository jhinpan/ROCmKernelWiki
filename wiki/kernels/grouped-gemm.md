---
id: kernel-grouped-gemm
title: "Grouped GEMM — Variable-Size Expert GEMMs in One Launch (MoE)"
type: kernel
architectures:
- gfx942
- gfx950
tags:
- grouped-gemm
- moe
- gemm
- mfma
- fp8
- bf16
- persistent-kernel
- stream-k
- tile-scheduling
confidence: source-reported
reproducibility: snippet
kernel_types:
- grouped-gemm
- gemm
- moe
languages:
- composable-kernel
- hip
related:
- kernel-fused-moe
- technique-stream-k
- technique-persistent-kernel
- hw-mfma
- kernel-ck-hgemm
- kernel-fp8-gemm
sources:
- ref-composable-kernel
- ref-aiter
- blog-cktile-gemm
- blog-gemm-optimization
- doc-mi300x-datasheet
performance_claims:
- gpu: MI300X
  dtype: bf16
  metric: peak-tflops
  value: "~950 TFLOPS (large balanced groups, compute-bound)"
  shape: "G=8 experts, M_g≈4096, N=K=8192"
  utilization: "~73% of 1307 TFLOPS dense BF16 peak"
  source_id: doc-mi300x-datasheet
  baseline: "looped per-expert hipBLASLt calls"
- gpu: MI300X
  dtype: fp8
  metric: speedup
  value: "1.3-1.8x vs per-expert GEMM loop at decode token counts"
  shape: "many small/uneven M_g, N=K=7168"
  source_id: ref-aiter
  baseline: "G separate hipBLASLt launches"
---

# Grouped GEMM — Variable-Size Expert GEMMs in One Launch (MoE)

## Overview

A **grouped GEMM** computes a *batch of independent matrix multiplies whose
inner dimensions differ per group* in a **single kernel launch**:

```
for g in 0..G-1:   C_g[M_g, N] = A_g[M_g, K] · B_g[K, N]
```

The defining feature versus a *batched* GEMM is that `M_g` (and sometimes `N_g`)
varies per group. This is exactly the shape produced by a **Mixture-of-Experts
(MoE)** layer: after the router scatters tokens to experts, expert `g` receives
`M_g` tokens, and the per-expert token counts are data-dependent and uneven
(often heavily skewed). Launching `G` separate GEMMs wastes the GPU — small
`M_g` underfills the [304 CUs of an MI300X](../hardware/chiplet-xcd.md), and the
fixed launch/epilogue overhead is paid `G` times. A grouped GEMM fuses all `G`
problems behind one grid so the scheduler can keep every CU busy regardless of
how lopsided the token distribution is.

Grouped GEMM is the GEMM half of a [fused MoE](fused-moe.md) layer; the gate/up
projection and the down projection are each a grouped GEMM (with SiLU/Swish
activation fused into the first one's epilogue).

## The scheduling problem

The core difficulty is that the (group, tile) work list is only known at
runtime, after the router produces `M_g`. Two scheduling styles dominate on
CDNA:

1. **Persistent + flattened tile index.** Launch exactly one workgroup per CU
   slot (a [persistent kernel](../techniques/persistent-kernel.md)) and have each
   workgroup pull tiles from a single global counter. A prefix-sum over the
   per-group tile counts maps a flat `tile_id` back to `(group, m_tile, n_tile)`.
   This naturally load-balances skewed `M_g` because idle workgroups simply grab
   the next available tile.
2. **Stream-K over the flattened space.** When many groups have tiny `M_g`, even
   tile-granular work can leave CUs idle on the K-reduction tail. Layering
   [Stream-K](../techniques/stream-k.md) on top splits the K loop across
   workgroups and reduces partials, smoothing the
   [tail effect](../patterns/tail-effect.md) for the long-tail experts.

Both rely on a host- or device-built **problem descriptor array** (the `M_g`
offsets, A/B/C base pointers, leading dimensions). In MoE the token→expert
permutation also produces a *sorted token index* so each tile reads a
contiguous slab of `A`.

## Host-side problem setup (Composable Kernel)

Composable Kernel (`ck::tensor_operation::device::DeviceGroupedGemm`) exposes
grouped GEMM directly. You fill a `std::vector` of per-group descriptors, then a
single `Run` issues one kernel. The kernel itself emits
[`v_mfma_*`](../hardware/mfma.md) for the inner tile.

```cpp
#include "ck/tensor_operation/gpu/device/impl/device_grouped_gemm_xdl.hpp"

using Row = ck::tensor_layout::gemm::RowMajor;
using F16 = ck::half_t;
using F32 = float;
using PassThrough = ck::tensor_operation::element_wise::PassThrough;

// One DeviceGroupedGemmXdl instance covers all G experts in a single launch.
using DeviceGroupedGemm = ck::tensor_operation::device::DeviceGroupedGemmXdl<
    Row, Row, Row, F16, F16, F16, F32,           // A,B,C layouts + dtypes, FP32 acc
    PassThrough, PassThrough, PassThrough,        // a,b,c elementwise ops
    /*GemmSpec*/ ck::tensor_operation::device::GemmSpecialization::MNKPadding,
    /*BlockSize*/ 256,
    /*MPerBlock*/ 256, /*NPerBlock*/ 128, /*KPerBlock*/ 32,
    /*MPerXDL*/ 32, /*NPerXDL*/ 32,               // 32x32x8 BF16/FP16 MFMA tile
    /*MXdlPerWave*/ 4, /*NXdlPerWave*/ 2>;

std::vector<ck::tensor_operation::device::GemmDesc> descs;
std::vector<const void*> a_ptrs, b_ptrs;
std::vector<void*>       c_ptrs;
for (int g = 0; g < G; ++g) {
    int Mg = host_tokens_per_expert[g];           // data-dependent, uneven
    descs.push_back({Mg, N, K, /*lda*/K, /*ldb*/N, /*ldc*/N});
    a_ptrs.push_back(A_sorted + token_offset[g] * K);
    b_ptrs.push_back(W + g * (size_t)K * N);       // expert g weight slab
    c_ptrs.push_back(Y + token_offset[g] * N);
}

DeviceGroupedGemm gemm;
auto arg = gemm.MakeArgument(a_ptrs, b_ptrs, {}, c_ptrs, descs,
                             PassThrough{}, PassThrough{}, PassThrough{});
// Workspace holds the flattened tile→(group,tile) map; build it on device.
gemm.SetWorkSpacePointer(&arg, workspace);
auto invoker = gemm.MakeInvoker();
invoker.Run(arg, StreamConfig{stream});           // <-- single launch, all G groups
```

The `MPerBlock/NPerBlock/KPerBlock` block tile and the `MPerXDL/NPerXDL` matrix
tile must be chosen so the accumulator fits in [AGPRs](../hardware/wavefront.md)
without spilling; `MXdlPerWave × NXdlPerWave` controls how many MFMA tiles each
wave accumulates and trades directly against occupancy. For FP8 experts swap the
dtypes to `ck::f8_t` (FNUZ on gfx942, OCP on gfx950) and use the `16x16x32` /
`32x32x16` FP8 MFMA shapes — see [FP8 GEMM](fp8-gemm.md).

## Device-side flattened tile scheduling (sketch)

When writing the scheduler by hand (e.g. in a HIP persistent kernel), the
essential loop maps a global tile counter onto the per-group tile space built
from a prefix sum of `ceil(M_g / MPerBlock)`:

```cpp
// grid = number of persistent workgroups (≈ #CUs); g_tile_counter in global mem.
__global__ void grouped_gemm_persistent(const GroupDesc* __restrict__ groups,
                                        const int* __restrict__ tile_prefix, // len G+1
                                        int total_tiles, int* g_tile_counter) {
  __shared__ int tile_id_s;
  for (;;) {
    if (threadIdx.x == 0) tile_id_s = atomicAdd(g_tile_counter, 1);
    __syncthreads();
    int tile_id = tile_id_s;
    if (tile_id >= total_tiles) return;            // drained: persistent exit

    // Binary search the prefix sum: which expert owns this flat tile?
    int g = upper_bound(tile_prefix, G + 1, tile_id) - 1;
    int local   = tile_id - tile_prefix[g];
    int n_tiles = (groups[g].N + NPerBlock - 1) / NPerBlock;
    int m_tile  = local / n_tiles;
    int n_tile  = local % n_tiles;

    // ... load A_g/B_g tiles via buffer_load (OOB-guarded for the M_g remainder),
    //     accumulate with v_mfma_* over K, write C_g.
    mfma_accumulate_tile(groups[g], m_tile, n_tile);
  }
}
```

[`buffer_load`](../hardware/memory-instructions.md) with its built-in
out-of-bounds semantics is the clean way to handle the ragged `M_g` remainder
tile: lanes past `M_g` read 0 instead of needing a scalar branch. The atomic
counter gives near-perfect load balance across the uneven groups at the cost of
one global atomic per tile (amortized over the whole MFMA inner loop, so it is
negligible).

## Practical notes

- **XCD locality.** On MI300X the L2 is [per-XCD](../patterns/xcd-locality.md). A
  pure atomic tile-pull scatters a group's tiles across all 8 XCDs, hurting
  weight reuse. Biasing the flat ordering so consecutive tiles of one expert land
  on the same XCD recovers L2 hit rate; AITER and CK both expose tuning here.
- **Weight layout.** Pre-shuffling `B_g` into the MFMA-friendly
  [preshuffle layout](../techniques/preshuffle-layout.md) removes an LDS
  transpose in the inner loop and is a large win when `M_g` is small and the
  kernel is weight-bandwidth-bound.
- **Decode vs prefill.** At decode (tiny total tokens) grouped GEMM is
  **memory-bound** — the win over a per-expert loop comes from launch-overhead
  amortization and CU occupancy, not FLOPS. At prefill (large `M_g`) it
  approaches dense GEMM efficiency.
- **Quantization.** Per-expert / per-token-group FP8 scales fold into the
  epilogue; see [fine-grained quantization](../techniques/fine-grained-quantization.md).

## Performance

Numbers are configuration-dependent; treat as order-of-magnitude. The
[MI300X datasheet](../../sources/docs/doc-mi300x-datasheet.md) gives dense peaks
of 1307 TFLOPS BF16 and 2615 TFLOPS FP8 — grouped GEMM with large, balanced
groups reaches a healthy fraction of BF16 peak, while skewed small-`M_g` decode
shapes are bound by HBM bandwidth (5.3 TB/s) and benefit most from the
single-launch scheduling versus a per-expert loop ([AITER](../../sources/refs/ref-aiter.md)).

## See also

- [Fused MoE kernel](fused-moe.md) — the layer this GEMM lives in
- [Stream-K](../techniques/stream-k.md) and [persistent kernels](../techniques/persistent-kernel.md)
- [MFMA matrix cores](../hardware/mfma.md), [FP8 GEMM](fp8-gemm.md)
- [Tail-effect pattern](../patterns/tail-effect.md)

## Sources

- [Composable Kernel — grouped GEMM device ops](https://github.com/ROCm/composable_kernel)
- [AITER — AMD AI operator library (FusedMoE / grouped GEMM)](https://github.com/ROCm/aiter)
- [CK-tile GEMM optimization blog](https://rocm.blogs.amd.com/)
- [Optimizing GEMM on AMD GPUs blog](https://rocm.blogs.amd.com/)
- [AMD Instinct MI300X datasheet](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/data-sheets/amd-instinct-mi300x-data-sheet.pdf)
