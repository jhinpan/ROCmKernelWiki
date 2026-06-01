---
id: hw-cross-lane
title: Cross-Lane Data Movement (DPP, ds_swizzle, ds_permute/bpermute, permlane)
type: hardware
version_sensitive:
- vs-permlane16-gfx950
architectures:
- gfx942
- gfx950
tags:
- dpp
- swizzle
- permute
- ds-instructions
- wave64
- wave-reduce
- dpp-reduction
confidence: source-reported
related:
- hw-wavefront
- hw-lds
- hw-mfma
- technique-wave-reduce
- lang-gcn-asm
sources:
- doc-cdna3-isa
- doc-cdna4-isa
- ref-gcnasm
- doc-llvm-amdgpu
- blog-amd-matrix-cores
- blog-amdgpu-kernel-opt-guide
aliases:
- cross-lane
- lane shuffle
- warp shuffle
- bpermute
- permlane16
- dpp
---
# Cross-Lane Data Movement (DPP, ds_swizzle, ds_permute/bpermute, permlane)

## Overview

Many kernels need lanes of a wavefront to read each other's registers without a
round-trip through HBM: reductions, broadcasts, transposes, butterfly shuffles,
softmax normalization across a row. CDNA exposes several mechanisms that move a
VGPR from a *source* lane to a *destination* lane, all without touching global
memory. They differ in **reach** (which lanes can talk to which), **cost**
(extra issue cycles, LDS-unit occupancy, wait states), and **flexibility**
(fixed pattern vs. data-dependent index):

| Mechanism | Reach | Index source | Uses LDS unit? | Notes |
|---|---|---|---|---|
| DPP modifier | within 16-lane row (some cross-row patterns) | fixed in opcode | no | free-ish ALU modifier; EXEC hazard |
| `ds_swizzle_b32` | within 32-lane group | fixed (encoded mask) | yes (no storage) | compile-time pattern |
| `ds_permute_b32` (fwd) | full 64 lanes | per-lane VGPR (push) | yes (no storage) | scatter: I send to lane *idx* |
| `ds_bpermute_b32` (bwd) | full 64 lanes | per-lane VGPR (pull) | yes (no storage) | gather: I read from lane *idx* |
| `v_permlane16_*` | 32-lane half, gfx950-only | SGPR selectors | no | absent on gfx942 |
| `v_readlane`/`v_writelane`/`v_readfirstlane` | lane ↔ SGPR | immediate / VGPR | no | vector↔scalar move |

CDNA is **wave64-only**, so all of these operate over a 64-lane wavefront and a
64-bit `EXEC` mask. This is the AMD analog of NVIDIA's `__shfl_sync` /
`__reduce_*` warp primitives; HIP's `__shfl*` family is implemented on top of
`ds_bpermute`/DPP. Query `warpSize` (64 on gfx9) — never hardcode 32.

## DPP — Data Parallel Primitives modifier

DPP is not a standalone instruction; it is a **modifier** attached to a normal
VALU op (`v_mov_b32`, `v_add_f32`, …) that, before the ALU executes, replaces
each lane's operand with a value pulled from a neighbor according to a fixed
pattern. Because the ALU does the work anyway, DPP shuffles are nearly free in
issue slots. Patterns operate on the wavefront viewed as **rows of 16 lanes**:
row shift/rotate (`row_shl`, `row_shr`, `row_ror`), full-row broadcast, mirror,
and (on CDNA) some cross-row broadcasts (`row_bcast:15`, `row_bcast:31`).

```cpp
// Tree reduction inside a 16-lane row using DPP add.
// __builtin_amdgcn_mov_dpp(src, dpp_ctrl, row_mask, bank_mask, bound_ctrl)
// dpp_ctrl 0x111 == row_shr:1, 0x112 == row_shr:2, ...
__device__ float row_reduce_add(float v) {
    for (int off = 1; off < 16; off <<= 1) {
        float n = __builtin_amdgcn_mov_dpp(
            __builtin_bit_cast(int, v),
            0x110 | off,   // row_shr by `off`
            0xf, 0xf,      // all 4 rows, all 4 banks active
            false);        // bound_ctrl: out-of-range lane => use own value
        v += n;
    }
    return v; // lane 0 of each row now holds the row sum
}
```

> **EXEC hazard.** The ISA notes that a VALU op does **not** forward its `EXEC`
> mask to a following DPP op; the compiler must insert wait states (NOPs)
> between a write of a register and its DPP consumption. With the builtin this is
> handled for you, but in hand assembly you must respect the documented
> `v_*_dpp` wait-state rules or you read stale lane data. `bound_ctrl` selects
> what an out-of-range neighbor contributes (0 vs. the lane's own value).

## ds_swizzle_b32 — fixed intra-group shuffle

`ds_swizzle_b32` routes one dword between lanes within **32-lane groups** using a
pattern encoded directly in the instruction (no index register). It runs on the
LDS unit but consumes **no LDS storage** and generates no bank conflicts. It is
ideal for fixed butterfly/quad patterns (e.g. the cross-lane step of an FFT or a
quad transpose) where the permutation is known at compile time. Lanes that map
to an invalid source contribute 0.

```cpp
// Swap adjacent pairs within each group of 4 lanes (quad swizzle).
// The 0x8000 "FFT/quad" mode encodes a fixed 5-bit pattern per lane.
__device__ int quad_swap(int v) {
    return __builtin_amdgcn_ds_swizzle(v, 0x041f); // QDMode: reverse within quads
}
```

## ds_permute / ds_bpermute — full-wave data-dependent gather/scatter

These two are the work-horses for **arbitrary 64-lane** movement with a
**runtime, per-lane index**. They reuse the LDS crossbar but, again, store
nothing in LDS:

- `ds_bpermute_b32` — **backward / pull / gather**: each lane provides a source
  lane id; it receives that lane's data. "Read from lane `idx`."
- `ds_permute_b32` — **forward / push / scatter**: each lane provides a
  destination lane id; it sends its data there. "Write my value to lane `idx`."

The index is a **byte address** = `lane_id * 4` (the hardware multiplies the lane
id by 4 to index dwords), so an index VGPR must hold `target_lane << 2`. On a
scatter collision (two lanes target the same destination) the **highest source
lane wins**; for `ds_bpermute` an out-of-range source yields 0.

```cpp
// Full 64-lane reduction: DPP within rows, then ds_bpermute across rows.
// idx in *bytes*: (target_lane << 2). Builtin takes the byte address directly.
__device__ float wave_reduce_add(float v) {
    // 1) reduce within each 16-lane row with DPP (see above)
    v = row_reduce_add(v);                 // lane{0,16,32,48} hold partials
    // 2) gather the 4 row-leaders into lane 0 via ds_bpermute
    int self = __builtin_amdgcn_ds_bpermute(0  << 2, __builtin_bit_cast(int,v));
    int r1   = __builtin_amdgcn_ds_bpermute(16 << 2, __builtin_bit_cast(int,v));
    int r2   = __builtin_amdgcn_ds_bpermute(32 << 2, __builtin_bit_cast(int,v));
    int r3   = __builtin_amdgcn_ds_bpermute(48 << 2, __builtin_bit_cast(int,v));
    float acc = __builtin_bit_cast(float,self) + __builtin_bit_cast(float,r1)
              + __builtin_bit_cast(float,r2)   + __builtin_bit_cast(float,r3);
    // 3) broadcast lane-0 result to the whole wave
    return __builtin_amdgcn_readfirstlane(__builtin_bit_cast(int, acc));
}
```

This DPP-then-`ds_bpermute`-then-`readfirstlane` sequence is the canonical
gfx942 wave reduction — see [wave reduction](../techniques/wave-reduce.md). Only
32-bit dwords move per op; 64-bit values need two passes.

## v_permlane16 — gfx950 only

CDNA4 adds `v_permlane16_b32` / `v_permlanex16_b32`, which permute lanes within a
16-lane block (and the cross-block `x16` variant) using **SGPR-held selectors**
rather than the LDS crossbar — useful for the cross-row step of a reduction
without occupying the LDS unit. These are **absent on gfx942**: code that uses
`v_permlane16_*` will not assemble for CDNA3, so a portable kernel must fall back
to `ds_bpermute`/DPP. This is one of the small ISA deltas to watch when porting —
see [gfx942 → gfx950 migration](../migration/gfx942-to-gfx950.md).

```cpp
#if defined(__gfx950__)
    // gfx950: cross-lane step without touching the LDS unit
    int hi = __builtin_amdgcn_permlanex16(v, v, 0x76543210u, 0xfedcba98u,
                                          /*fi=*/true, /*bound_ctrl=*/false);
#else
    // gfx942 fallback: pull from the partner lane via the LDS crossbar
    int hi = __builtin_amdgcn_ds_bpermute((lane ^ 16) << 2, v);
#endif
```

## readlane / writelane — vector ↔ scalar

`v_readlane_b32` copies one lane's VGPR into an SGPR; `v_readfirstlane_b32` reads
the lowest active lane (handy for uniformizing a value so it can drive scalar
control flow or an SGPR operand); `v_writelane_b32` writes an SGPR into a chosen
lane. These move data across the vector/scalar boundary rather than between
lanes, and are the usual final step of a reduction (broadcast the lane-0 result)
or the way to feed a per-wave scalar into a `buffer`/`ds` address.

## Choosing a primitive

- Fixed pattern, all within a 16-lane row → **DPP** (cheapest).
- Fixed pattern within 32 lanes → **ds_swizzle_b32**.
- Data-dependent or full-64-lane movement → **ds_permute/ds_bpermute**.
- gfx950 cross-row step without the LDS unit → **v_permlane16/x16**.
- Lane → scalar (broadcast / uniformize) → **readfirstlane/readlane**.

`ds_swizzle`, `ds_permute`, and `ds_bpermute` issue on the LDS unit and so
contend with real `ds_read`/`ds_write` traffic; on a heavily LDS-bound kernel a
DPP or `v_permlane16` path can be cheaper even when its reach is narrower.

## Measured latency (MI300) and how to choose

The nod-ai/shark-ai *AMDGPU Kernel Optimization Guide* reports measured
per-primitive latencies on MI300 (Fused Softmax microbenchmark; cycles include
the instruction **plus its `s_waitcnt`**):

| Primitive | Approx. cycles | Needs `s_waitcnt`? | Reach |
|---|---|---|---|
| `ds_permute` / `ds_bpermute` | ~50 | yes (LDS unit) | full 64-lane, arbitrary |
| `ds_swizzle` | ~50 | yes (LDS unit) | fixed pattern, 32-lane groups |
| DPP | 4–12 | no | adjacent rows / fixed shifts |
| `v_permlane` (gfx950) | 4–8 | no | 16/32-lane, gfx950 only |

The guide's rule of thumb — **speed:**
`v_permlane ≥ DPP > ds_swizzle ≥ ds_permute > ds_bpermute`; **generality** is the
exact reverse. Practical guidance: reach for **DPP** (or `v_permlane16` on gfx950)
whenever the access pattern fits — it is ~5–10× cheaper than the LDS-crossbar ops
and needs no `s_waitcnt` — and fall back to `ds_permute`/`ds_bpermute` only when
you need arbitrary full-wave gather/scatter. In MLIR these surface as
`amdgpu.dpp` / `rocdl.update.dpp`, `rocdl.ds_swizzle`, `rocdl.ds_bpermute`, and
`rocdl.permlane*` / `amdgpu.permlane_swap`.

## Sources

- [CDNA3 ISA Reference Guide](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf)
- [CDNA4 ISA Reference Guide](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-cdna4-instruction-set-architecture.pdf)
- [LLVM AMDGPU User Guide — cross-lane intrinsics](https://llvm.org/docs/AMDGPUUsage.html)
- [GCN/CDNA assembly notes (gcnasm)](https://github.com/carlushuang/gcnasm)
- [AMD Matrix Cores blog](https://gpuopen.com/learn/amd-lab-notes/amd-lab-notes-matrix-cores-readme/)
- [AMDGPU Kernel Optimization Guide (nod-ai/shark-ai)](https://github.com/nod-ai/amd-shark-ai/blob/main/docs/amdgpu_kernel_optimization_guide.md) — measured cross-lane latencies
