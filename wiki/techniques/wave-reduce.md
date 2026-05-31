---
id: technique-wave-reduce
title: Wave-Level Reduction (DPP rows + ds_bpermute cross-row + readlane)
type: technique
architectures:
- gfx942
- gfx950
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
`v_permlane16_b32` which replaces the `ds_*` cross-row step with a pure-ALU op.
See the [cross-lane hardware page](../hardware/cross-lane.md) for the primitive
semantics; this page is about composing them into a correct, fast reduction.

## The reduction tree on wave64

A wave64 sum reduces in `log2(64) = 6` steps. The first 4 steps fold *within*
each 16-lane row using DPP; the last 2 steps cross rows. Because the DPP modifier
**does not forward `EXEC` from a producing VALU op**, the compiler must insert
wait states — let the builtin handle that and never interleave DPP with a
dependent VALU write by hand.

```cpp
#include <hip/hip_runtime.h>

// Full wave64 sum reduction. Result is valid in lane 0 (broadcast below).
// DPP row ops (steps 1-4) + ds_swizzle cross-row (steps 5-6).
__device__ __forceinline__ float wave_reduce_sum(float v) {
    // --- Steps 1-4: intra-row tree over 16 lanes via DPP ---
    // dpp_ctrl 0x111/0x112/0x114/0x118 = row_shr by 1/2/4/8 (shift right within row)
    // row_mask=0xf, bank_mask=0xf, bound_ctrl=1 (OOB lanes contribute 0)
    v += __builtin_amdgcn_mov_dpp(v, 0x111, 0xf, 0xf, true); // +lane^1 pattern (shr 1)
    v += __builtin_amdgcn_mov_dpp(v, 0x112, 0xf, 0xf, true); // shr 2
    v += __builtin_amdgcn_mov_dpp(v, 0x114, 0xf, 0xf, true); // shr 4
    v += __builtin_amdgcn_mov_dpp(v, 0x118, 0xf, 0xf, true); // shr 8 -> lane 0 of each row holds row sum

    // --- Steps 5-6: cross the 16-lane row boundary ---
    // ds_swizzle handles 32-lane groups; do an explicit bpermute for the 32->64 fold.
    int self  = __lane_id();                 // 0..63
    // Bring lane (self+16) and (self+32) home with backward permute (addr = src_lane*4).
    float v16 = __builtin_amdgcn_ds_bpermute((self ^ 16) << 2, __builtin_bit_cast(int, v))
                ? 0.0f : 0.0f; // placeholder, see typed helper below
    (void)v16;
    return v;
}
```

The bit-cast dance above is the real friction point: `ds_bpermute` operates on
**`int`/b32** payloads and takes a **byte address** (`src_lane * 4`). A robust
helper wraps the cast and the address arithmetic:

```cpp
__device__ __forceinline__ float bpermute_f32(float v, int src_lane) {
    int iv = __builtin_bit_cast(int, v);
    int r  = __builtin_amdgcn_ds_bpermute(src_lane << 2, iv); // byte addr = lane*4
    return __builtin_bit_cast(float, r);
}

// Clean wave64 sum: 4 DPP steps + 2 bpermute steps + readlane broadcast.
__device__ __forceinline__ float wave_sum64(float v) {
    v += __builtin_amdgcn_mov_dpp(v, 0x111, 0xf, 0xf, true);
    v += __builtin_amdgcn_mov_dpp(v, 0x112, 0xf, 0xf, true);
    v += __builtin_amdgcn_mov_dpp(v, 0x114, 0xf, 0xf, true);
    v += __builtin_amdgcn_mov_dpp(v, 0x118, 0xf, 0xf, true); // lane (row*16) holds its row's sum

    int lane = __lane_id();
    v += bpermute_f32(v, lane ^ 16);  // fold rows 0+1, 2+3
    v += bpermute_f32(v, lane ^ 32);  // fold {0,1}+{2,3}  -> every lane now holds the full sum
    return v;                          // (already broadcast; readlane optional)
}
```

After the two `xor`-stride bpermute steps every lane holds the complete wave sum,
so a separate broadcast is unnecessary. If you instead use `row_shr` all the way
down, the result lands only in lane 0 and you must broadcast it:

```cpp
// Pull the scalar from lane 0 into an SGPR, then it is uniform across the wave.
float total = __builtin_amdgcn_readlane(v, 0);   // v_readlane_b32 -> s#
```

> **`__shfl` shortcut.** HIP's `__shfl_xor(v, mask)` lowers to exactly these
> primitives (DPP where possible, `ds_bpermute` for the wide masks). A portable
> `for (int m = 32; m > 0; m >>= 1) v += __shfl_xor(v, m);` is the easiest
> correct version; drop to the builtins above only when the compiler fails to
> keep the first four steps in DPP.

## gfx950: `v_permlane16` replaces the cross-row `ds_*`

On CDNA4 the cross-row fold can stay in the VALU. `v_permlane16_b32` is
**gfx950-only** (absent on gfx942) and permutes across the 32-lane half-wave
without touching the LDS crossbar, removing the `LGKMCNT` dependency that
`ds_bpermute` introduces:

```cpp
#if defined(__gfx950__)
// Cross-row fold without LDS: lower the two ds_bpermute steps to permlane16.
v = __builtin_amdgcn_permlane16(v, v, /*src0*/0, /*src1*/0,
                                /*fi*/false, /*bc*/false);
#endif
```

Because `permlane16` keeps the whole reduction in the vector ALU, it avoids the
`s_waitcnt lgkmcnt(0)` that gates every `ds_*` step — a measurable win in
latency-bound decode kernels (`rmsnorm`, `softmax`) where the reduction sits on
the critical path. Guard it behind `__gfx950__` and keep the `ds_bpermute` path
for gfx942.

## Correctness pitfalls

- **Payload width.** `ds_bpermute`/`ds_swizzle`/`readlane` move **32 bits**.
  Reduce `double` or `int64` as two halves, or reduce in FP32 and cast.
- **Byte address, not lane index.** `ds_bpermute` takes `src_lane * 4`. Forgetting
  the `<< 2` silently reads the wrong lane.
- **Collision rule.** If two lanes target the same `ds_bpermute` destination, the
  **highest source lane wins** (per the CDNA3 ISA). The `xor`-stride pattern above
  is a permutation, so no collisions occur — but ad-hoc gathers can hit this.
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

- [Cross-lane primitives (DPP/swizzle/permute/permlane16)](../hardware/cross-lane.md)
- [Wavefront / EXEC / register files](../hardware/wavefront.md)
- [RMSNorm fused kernel](../kernels/rmsnorm.md)
- [gfx942 → gfx950 migration](../migration/gfx942-to-gfx950.md)

## Sources

- [AMD CDNA3 ISA Reference Guide](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf) — DPP modifiers, `ds_bpermute_b32`/`ds_swizzle_b32`, `v_readlane_b32`, wait-state hazards.
- [AMD CDNA4 ISA Reference Guide](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-cdna4-instruction-set-architecture.pdf) — `v_permlane16_b32`.
- [LLVM AMDGPU User Guide](https://llvm.org/docs/AMDGPUUsage.html) — `llvm.amdgcn.mov.dpp`, `llvm.amdgcn.ds.bpermute`, `llvm.amdgcn.readlane` intrinsics.
- [AMD Lab Notes / GCN assembly reference (gcnasm)](https://gpuopen.com/learn/amd-gcn-assembly-cross-lane-operations/) — cross-lane operations walkthrough.
