---
id: technique-wave-reduce
title: Wave-Level Reduction (DPP rows + ds_bpermute cross-row + readlane)
type: technique
architectures:
- gfx942
- gfx950
version_sensitive:
- vs-permlane16-gfx950
tags:
- wave-reduce
- dpp-reduction
- dpp
- permute
- swizzle
- wave64
- reduction
- cross-lane
confidence: source-reported
reproducibility: snippet
hardware_features:
- dpp
- permute
- swizzle
- wave64
- ds-instructions
kernel_types:
- reduction
- rmsnorm
- softmax
- layernorm
languages:
- hip
- gcn-asm
related:
- hw-cross-lane
- hw-wavefront
- hw-lds
- kernel-rmsnorm
- technique-kernel-fusion
sources:
- hw-cross-lane
- doc-cdna3-isa
- doc-cdna4-isa
- doc-llvm-amdgpu
- ref-gcnasm
- blog-amdgpu-kernel-opt-guide
implemented_by:
- pr-FlyDSL-447
- pr-FlyDSL-524
- pr-composable_kernel-2594
- pr-FlyDSL-450
- pr-FlyDSL-300
- pr-sglang-23882
- pr-composable_kernel-3259
- pr-composable_kernel-2978
---
# Wave-Level Reduction (DPP rows + ds_bpermute cross-row + readlane)

## Overview

A *wave reduction* collapses one value per lane into a single scalar (sum, max,
min, …) across an entire **wave64** wavefront. It is the inner step of almost
every normalization and attention kernel: `rmsnorm` and `layernorm` need a
sum-of-squares, `softmax`/`flash-attention` need a row max and a row sum, and
GEMV/`all-reduce` epilogues need partial-sum folding.

On CDNA there is **no single-instruction warp reduce**. You build the tree from
three cross-lane primitives, in increasing cost:

1. **DPP** (`v_mov_b32_dpp` / `__builtin_amdgcn_mov_dpp`) — shift/rotate/broadcast
   *within a 16-lane row*. Cheapest: stays in the VALU, no LDS traffic.
2. **`ds_swizzle_b32` / `ds_bpermute_b32`** — cross the 16-lane row boundary
   (lanes 0–15 ↔ 16–31 ↔ 32–47 ↔ 48–63) through the LDS crossbar, **without
   using any LDS storage**.
3. **`v_readlane_b32`** — extract the final scalar from lane 0 into an SGPR.

This is the canonical pattern documented for gfx942; gfx950 adds
`v_permlane16_swap_b32` / `v_permlane32_swap_b32`, which can replace compatible
`ds_*` cross-row exchanges with pure-VALU swaps. See the
[cross-lane hardware page](../hardware/cross-lane.md) for the exact primitive
semantics; this page is about composing them into a correct reduction.

## A correct wave64 baseline

A wave64 butterfly needs `log2(64) = 6` exchanges. HIP's `__shfl_xor` is the
safest starting point because it preserves the butterfly invariant across all
six masks and lets the backend choose target-appropriate DPP/cross-lane ops:

```cpp
#include <hip/hip_runtime.h>

// Requires all 64 lanes to participate; every lane receives the full sum.
__device__ __forceinline__ float wave_sum64(float v) {
    #pragma unroll
    for (int mask = 32; mask > 0; mask >>= 1)
        v += __shfl_xor(v, mask, 64);
    return v;
}
```

For a partially active wave, first define which inactive lanes contribute and
predicate them to the identity value. Inspect final ISA before assuming which
shuffle masks became DPP rather than LDS-crossbar operations.

## Explicit DPP row reduction plus full-wave gather

When hand-scheduling DPP, two details are easy to get wrong: the control must be
an immediate, and `row_shr` moves the complete 16-lane result to the row's
**last** lane (`15`, `31`, `47`, `63`). `bound_ctrl=true` makes an invalid source
contribute zero. The b32 intrinsics also require bit casts, not numeric
int/float conversions:

```cpp
__device__ __forceinline__ float dpp_row_sum_at_end(float v) {
    int n = __builtin_amdgcn_mov_dpp(__builtin_bit_cast(int, v),
                                     0x111, 0xf, 0xf, true);
    v += __builtin_bit_cast(float, n);
    n = __builtin_amdgcn_mov_dpp(__builtin_bit_cast(int, v),
                                 0x112, 0xf, 0xf, true);
    v += __builtin_bit_cast(float, n);
    n = __builtin_amdgcn_mov_dpp(__builtin_bit_cast(int, v),
                                 0x114, 0xf, 0xf, true);
    v += __builtin_bit_cast(float, n);
    n = __builtin_amdgcn_mov_dpp(__builtin_bit_cast(int, v),
                                 0x118, 0xf, 0xf, true);
    v += __builtin_bit_cast(float, n);
    return v;
}

__device__ __forceinline__ float bpermute_f32(float v, int src_lane) {
    int r = __builtin_amdgcn_ds_bpermute(src_lane << 2,
                                         __builtin_bit_cast(int, v));
    return __builtin_bit_cast(float, r);
}

// Correctness-oriented explicit path: every lane gathers all four row ends.
__device__ __forceinline__ float wave_sum64_dpp(float v) {
    v = dpp_row_sum_at_end(v);
    return bpermute_f32(v, 15) + bpermute_f32(v, 31)
         + bpermute_f32(v, 47) + bpermute_f32(v, 63);
}
```

This explicit path uses four gathers per lane and is a clarity reference, not a
claim of optimal scheduling. More selective leader-lane schemes can reduce
crossbar work, but must preserve the row-end locations and then broadcast a
correctly bit-cast result.

## gfx950: use `v_permlane*_swap`, not the RDNA selector form

CDNA4's relevant instructions are `v_permlane16_swap_b32` and
`v_permlane32_swap_b32`, exposed as
`__builtin_amdgcn_permlane16_swap` / `permlane32_swap`. They return a two-element
vector corresponding to the swapped first and second input registers; with the
same input passed twice, the partner is element 1 in the lower half and element
0 in the upper half. They take only `fi` and `bound_ctrl` booleans. The
selector-form `__builtin_amdgcn_permlane16` / `permlanex16` needs the RDNA
`gfx10-insts` feature and does **not** compile for gfx950.

If a row partial has already been replicated to every lane in its 16-lane row,
the two cross-row butterfly steps can stay in the VALU:

```cpp
#if defined(__gfx950__)
int lane = __lane_id();
int bits = __builtin_bit_cast(int, row_partial_replicated);
auto p16 = __builtin_amdgcn_permlane16_swap(bits, bits, false, false);
int partner16 = (lane & 16) ? p16[0] : p16[1];
float v = row_partial_replicated + __builtin_bit_cast(float, partner16);
bits = __builtin_bit_cast(int, v);
auto p32 = __builtin_amdgcn_permlane32_swap(bits, bits, false, false);
int partner32 = (lane & 32) ? p32[0] : p32[1];
v += __builtin_bit_cast(float, partner32);
#endif
```

That precondition matters: the DPP `row_shr` example above leaves a valid sum
only at each row end, so use the explicit gathers (or replicate the row partial)
before applying the swap/add form. The swaps avoid LDS-unit traffic, but their
latency benefit remains architecture/workload-specific and should be measured.

## Correctness pitfalls

- **Payload width.** `ds_bpermute`/`ds_swizzle`/`readlane` move **32 bits**.
  Reduce `double` or `int64` as two halves, or reduce in FP32 and cast.
- **Byte address, not lane index.** `ds_bpermute` takes `src_lane * 4`. Forgetting
  the `<< 2` silently reads the wrong lane.
- **Direction-specific collision rule.** `ds_bpermute` is a pull/gather: each
  destination lane independently names one source, so there is no multi-writer
  destination collision. The “highest source lane wins” rule belongs to the
  push/scatter `ds_permute` when several sources name one destination.
- **Inactive lanes.** With a partial `EXEC` mask, disabled source lanes return 0
  for `ds_bpermute` and `ds_swizzle` returns 0 for invalid lanes. Use
  `bound_ctrl=1` on DPP so out-of-range row lanes contribute the additive
  identity (0). For `max`/`min`, seed inactive lanes with `-INF`/`+INF` instead.
- **DPP EXEC hazard.** The VALU does not forward `EXEC` to a DPP read; the
  builtin inserts the required wait states, but hand-written assembly must add
  `s_nop` / honor the documented wait-state count (see the
  [CDNA3 ISA](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf)).

## Why not LDS?

A naive reduction writes 64 values to LDS, `s_barrier`, then has lane 0 read them
back — costing a barrier and LDS bank traffic. The cross-lane tree uses **zero
LDS storage** (`ds_bpermute` uses the LDS *crossbar* only) and **no barrier**,
so it composes cleanly inside a fused kernel where the wave already holds its
partials in registers. For block-level reductions across multiple waves you
still need one LDS round-trip — do the *wave* reduction with this technique
first, write one scalar per wave to LDS, then reduce those (see
[LDS](../hardware/lds.md) and [kernel fusion](kernel-fusion.md)).

## See also

- [Cross-lane primitives (DPP/swizzle/permute/permlane swap)](../hardware/cross-lane.md)
- [Wavefront / EXEC / register files](../hardware/wavefront.md)
- [RMSNorm fused kernel](../kernels/rmsnorm.md)
- [gfx942 → gfx950 migration](../migration/gfx942-to-gfx950.md)

## Sources

- [AMD CDNA3 ISA Reference Guide](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf) — DPP modifiers, `ds_bpermute_b32`/`ds_swizzle_b32`, `v_readlane_b32`, wait-state hazards.
- [AMD CDNA4 ISA Reference Guide](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-cdna4-instruction-set-architecture.pdf) — `v_permlane16_swap_b32` / `v_permlane32_swap_b32`.
- [LLVM AMDGPU User Guide](https://llvm.org/docs/AMDGPUUsage.html) — `llvm.amdgcn.mov.dpp`, `llvm.amdgcn.ds.bpermute`, `llvm.amdgcn.readlane` intrinsics.
- [AMD Lab Notes / GCN assembly reference (gcnasm)](https://gpuopen.com/learn/amd-gcn-assembly-cross-lane-operations/) — cross-lane operations walkthrough.
- [AMDGPU Kernel Optimization Guide (captured snapshot)](https://github.com/nod-ai/amd-shark-ai/blob/efa471aeef66a260c85983cc41e833bfa769dade/docs/amdgpu_kernel_optimization_guide.md)
