---
id: hw-cross-lane
title: Cross-Lane Data Movement (DPP, ds_swizzle, ds_permute/bpermute, permlane)
type: hardware
version_sensitive:
- vs-permlane16-gfx950
- vs-mlir-dpp-combine-llvm20
- vs-ds-bpermute-address-cdna3-cdna4
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
| `v_permlane16_swap` / `v_permlane32_swap` | swaps the paired 16-/32-lane halves, gfx950-only | `fi` / `bound_ctrl` bools (no selectors) | no | absent on gfx942; the RDNA selector-form `v_permlane16`/`permlanex16` is a *different* gfx10+ op that will not assemble for CDNA4 |
| `v_readlane`/`v_writelane`/`v_readfirstlane` | lane ↔ SGPR | immediate / VGPR | no | vector↔scalar move |

CDNA is **wave64-only**, so all of these operate over a 64-lane wavefront and a
64-bit `EXEC` mask. This is the AMD analog of NVIDIA's `__shfl_sync` /
`__reduce_*` warp primitives; HIP's `__shfl*` family is implemented on top of
`ds_bpermute`/DPP. Query `warpSize` (64 on gfx9) — never hardcode 32.

Architecture manuals use different DPP names and encodings: **CDNA calls the
VALU modifier DPP**, while **RDNA documents DPP8/DPP16** forms. The programming
idea overlaps, but selectors, masks, and supported opcodes are target-specific;
do not copy an RDNA DPP8/DPP16 encoding into a CDNA kernel.

## DPP — Data Parallel Primitives modifier

DPP is not a standalone instruction; it is a **modifier** attached to a normal
VALU op (`v_mov_b32`, `v_add_f32`, …) that, before the ALU executes, replaces
each lane's operand with a value pulled from a neighbor according to a fixed
pattern. Because the ALU does the work anyway, DPP shuffles are nearly free in
issue slots. Patterns operate on the wavefront viewed as **rows of 16 lanes**:
row shift/rotate (`row_shl`, `row_shr`, `row_ror`), full-row broadcast, mirror,
and (on CDNA) some cross-row broadcasts (`row_bcast:15`, `row_bcast:31`).

```cpp
// Tree reduction inside a 16-lane row using DPP shifts.
// __builtin_amdgcn_mov_dpp(src, dpp_ctrl, row_mask, bank_mask, bound_ctrl)
// dpp_ctrl 0x111 == row_shr:1, 0x112 == row_shr:2, ...
__device__ float row_reduce_add(float v) {
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
    return v; // lanes 15,31,47,63 now hold their 16-lane row sums
}
```

> **EXEC hazard.** The ISA notes that a VALU op does **not** forward its `EXEC`
> mask to a following DPP op; the compiler must insert wait states (NOPs)
> between a write of a register and its DPP consumption. With the builtin this is
> handled for you, but in hand assembly you must respect the documented
> `v_*_dpp` wait-state rules or you read stale lane data. With these row shifts,
> `bound_ctrl=true` makes an invalid source contribute zero; `false` suppresses
> the update and preserves the prior destination value. A `row_shr` accumulation
> leaves the complete row result in the **last** lane, not lane 0.

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

The index is a **byte address** = `lane_id * 4`, so an index VGPR normally holds
`target_lane << 2`. The crossbar selects `((address + offset) / 4) % 64`, so
high address bits wrap rather than providing lane-index OOB protection. A
`ds_bpermute` read returns 0 when its selected source lane is disabled in
`EXEC`. On a `ds_permute` scatter collision (two lanes target the same
destination), the **highest source lane wins**.

```cpp
// Full 64-lane reduction: DPP within rows, then ds_bpermute across rows.
// idx in *bytes*: (target_lane << 2). Builtin takes the byte address directly.
__device__ float wave_reduce_add(float v) {
    // 1) reduce within each 16-lane row with DPP (see above)
    v = row_reduce_add(v);                 // lanes 15,31,47,63 hold partials
    int bits = __builtin_bit_cast(int, v);
    // 2) every lane gathers and adds the four valid row-end partials
    int r0 = __builtin_amdgcn_ds_bpermute(15 << 2, bits);
    int r1 = __builtin_amdgcn_ds_bpermute(31 << 2, bits);
    int r2 = __builtin_amdgcn_ds_bpermute(47 << 2, bits);
    int r3 = __builtin_amdgcn_ds_bpermute(63 << 2, bits);
    float acc = __builtin_bit_cast(float, r0) + __builtin_bit_cast(float, r1)
              + __builtin_bit_cast(float, r2) + __builtin_bit_cast(float, r3);
    // 3) uniformize via an SGPR; bit-cast instead of numerically converting int
    int scalar_bits = __builtin_amdgcn_readfirstlane(
        __builtin_bit_cast(int, acc));
    return __builtin_bit_cast(float, scalar_bits);
}
```

This DPP-then-`ds_bpermute`-then-`readfirstlane` sequence is the canonical
gfx942 wave reduction — see [wave reduction](../techniques/wave-reduce.md). Only
32-bit dwords move per op; 64-bit values need two passes.

## v_permlane16_swap — gfx950 only

CDNA4 adds the lane-**swap** instructions `v_permlane16_swap_b32` and
`v_permlane32_swap_b32`, which *exchange* the paired 16-/32-lane halves of the
wavefront in a single op without touching the LDS crossbar — useful for the
cross-row step of a reduction without occupying the LDS unit. They are reached
via `__builtin_amdgcn_permlane16_swap` / `__builtin_amdgcn_permlane32_swap`,
which return the two destination registers produced by swapping their two input
registers. Element 0 is the swapped first input and element 1 is the swapped
second input; neither element is universally “self” or “partner.” With the same
value passed twice, select element 0 in the upper 16-/32-lane half and element 1
in the lower half to obtain the partner. The builtins take only `fi` /
`bound_ctrl` boolean modifiers — there are **no SGPR selector operands**. These
are **absent on gfx942** (a portable kernel must fall back to
`ds_bpermute`/DPP there).

> **Watch the name.** The RDNA selector-form `v_permlane16_b32` /
> `v_permlanex16_b32` (`__builtin_amdgcn_permlane16` / `permlanex16`, taking two
> 32-bit SGPR selectors) is a **gfx10+ (RDNA) instruction**, *not* the CDNA4 op.
> On gfx950 the compiler rejects it with `error: '__builtin_amdgcn_permlanex16'
> needs target feature gfx10-insts` (verified on MI350X, ROCm 7.2 / clang 22).
> Use the `_swap` builtins on CDNA4. This is one of the small ISA deltas to watch
> when porting — see [gfx942 → gfx950 migration](../migration/gfx942-to-gfx950.md).

```cpp
#if defined(__gfx950__)
    // gfx950: swap the two 16-lane halves of each 32-lane group with
    // v_permlane16_swap_b32 — no LDS-unit traffic. The partner is in a
    // different result element in the lower and upper 16-lane halves.
    // (The RDNA selector-form permlanex16(...) does NOT assemble for CDNA4.)
    auto sw = __builtin_amdgcn_permlane16_swap(v, v, /*fi=*/false, /*bound_ctrl=*/false);
    int hi = (lane & 16) ? sw[0] : sw[1];
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
- gfx950 cross-row step without the LDS unit → **v_permlane16_swap** (lane-block swap).
- Lane → scalar (broadcast / uniformize) → **readfirstlane/readlane**.

`ds_swizzle`, `ds_permute`, and `ds_bpermute` issue on the LDS unit and so
contend with real `ds_read`/`ds_write` traffic; on a heavily LDS-bound kernel a
DPP or `v_permlane16` path can be cheaper even when its reach is narrower.

## Guide-reported latency estimates and how to choose

The nod-ai/shark-ai *AMDGPU Kernel Optimization Guide* labels this table as a
Fused Softmax measurement on MI300 using `rocprofv2`; cycles for LDS-crossbar
operations include the instruction **plus its `s_waitcnt`**. However, the same
table includes `v_permlane`, while the guide's own `v_permlane` section says it
is a CDNA4 TODO and gfx942 does not have the gfx950 `_swap` instructions. The
first three rows can therefore be retained as MI300 guide measurements; the
`v_permlane` row is only an architecture-scoped estimate of unclear provenance,
not an MI300 measurement or a local verification result.

| Primitive | Approx. cycles | Needs `s_waitcnt`? | Reach | Evidence scope |
|---|---:|---|---|---|
| `ds_permute` / `ds_bpermute` | ~50 | yes (LDS unit) | full 64-lane, arbitrary | guide-reported MI300 measurement |
| `ds_swizzle` | ~50 | yes (LDS unit) | fixed pattern, 32-lane groups | guide-reported MI300 measurement |
| DPP | 4–12 | no | adjacent rows / fixed shifts | guide-reported MI300 measurement |
| `v_permlane` (gfx950) | 4–8 | no | 16/32-lane, gfx950 only | guide estimate; cannot be from the stated MI300 instruction set |

The guide's unverified rule of thumb — **speed:**
`v_permlane ≥ DPP > ds_swizzle ≥ ds_permute > ds_bpermute`; **generality** is the
exact reverse. Practical guidance: reach for **DPP** (or `v_permlane16` on gfx950)
whenever the access pattern fits (on gfx950 use `v_permlane16_swap` for the
cross-row step). The guide's MI300 numbers make DPP roughly 4–12× lower latency
than its LDS-crossbar measurements and it needs no `s_waitcnt`; fall back to
`ds_permute`/`ds_bpermute` when
you need arbitrary full-wave gather/scatter. In MLIR these surface as
`amdgpu.dpp` / `rocdl.update.dpp`, `rocdl.ds_swizzle`, `rocdl.ds_bpermute`, and
`rocdl.permlane*` / `amdgpu.permlane_swap`.

## MLIR lowering and DPP fusion

The guide's MLIR mapping is tied to its LLVM 20-era snapshot:

| Hardware operation | MLIR operation | Important qualification |
|---|---|---|
| `ds_bpermute_b32` | `rocdl.ds_bpermute` | direct LDS-crossbar operation; requires an LDS wait |
| `ds_swizzle_b32` | `rocdl.ds_swizzle` | pattern encoded in the op; requires an LDS wait |
| `ds_permute_b32` | no dedicated `rocdl.ds_permute` in the captured snapshot | re-check when changing LLVM/MLIR revisions |
| DPP | `rocdl.update.dpp`, or the enum-friendly `amdgpu.dpp` wrapper | represented as a DPP move before backend combining |
| gfx950 lane swap | `rocdl.permlane*`, `amdgpu.permlane_swap` | exact op depends on the lane-swap form and MLIR revision |

DPP is a modifier on a VALU source, while the generic MLIR operation can be
represented initially as `v_mov_b32_dpp`. LLVM's `GCNDPPCombine` then makes a
**best-effort** attempt to fold that move into a compatible following VALU op:

```text
v_mov_b32_dpp + v_add_f32_e32  ->  v_add_f32_dpp   (when legal)
```

This fusion is not guaranteed. In the captured combine implementation,
non-default `row_mask` or `bank_mask` values (`!= 0xf`) are among the reasons a
move cannot be folded. Other operand, opcode, liveness, and modifier constraints
also apply. Inspect the final ISA instead of counting MLIR operations and
assuming a fused DPP instruction was selected. The exact historical condition
is visible in
[`GCNDPPCombine::combineDPPMov`](https://github.com/llvm/llvm-project/blob/ab51eccf88f5321e7c60591c5546b254b6afab99/llvm/lib/Target/AMDGPU/GCNDPPCombine.cpp#L522).

## Sources

- [CDNA3 ISA Reference Guide](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf)
- [CDNA4 ISA Reference Guide](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-cdna4-instruction-set-architecture.pdf)
- [LLVM AMDGPU User Guide — cross-lane intrinsics](https://llvm.org/docs/AMDGPUUsage.html)
- [GCN/CDNA assembly notes (gcnasm)](https://github.com/carlushuang/gcnasm)
- [AMD Matrix Cores blog](https://gpuopen.com/learn/amd-lab-notes/amd-lab-notes-matrix-cores-readme/)
- [AMDGPU Kernel Optimization Guide (captured snapshot)](https://github.com/nod-ai/amd-shark-ai/blob/efa471aeef66a260c85983cc41e833bfa769dade/docs/amdgpu_kernel_optimization_guide.md) — measured cross-lane latencies and LLVM/MLIR lowering notes
- [`rocprofv2`/rocprofiler link cited by the captured guide](https://github.com/ROCm/rocprofiler?tab=readme-ov-file#plugin-support)
- [MLIR ROCDL dialect reference](https://mlir.llvm.org/docs/Dialects/ROCDLDialect/)
- [MLIR AMDGPU dialect reference](https://mlir.llvm.org/docs/Dialects/AMDGPU/)
