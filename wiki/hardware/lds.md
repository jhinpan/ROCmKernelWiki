---
id: hw-lds
title: "LDS — Local Data Share (CDNA3/CDNA4 Shared Memory)"
type: hardware
architectures:
- gfx942
- gfx950
tags:
- lds
- ds-instructions
- bank-conflict-avoidance
- swizzle
- permute
- cdna
confidence: source-reported
related:
- hw-mfma
- hw-async-copy-lds
- hw-cross-lane
- technique-bank-conflict-avoidance
- technique-lds-double-buffering
- technique-lds-swizzling
- pattern-bank-conflicts
sources:
- doc-cdna3-isa
- doc-cdna4-isa
- blog-gemm-optimization
- doc-llvm-amdgpu
aliases:
- LDS
- "local data share"
- "shared memory"
- "groupshared"
- __shared__
---

# LDS — Local Data Share (CDNA3/CDNA4 Shared Memory)

## Overview

The **Local Data Share (LDS)** is the on-chip, software-managed scratchpad
private to each Compute Unit (CU). It is AMD's equivalent of NVIDIA *shared
memory*: low-latency, explicitly addressed SRAM used to stage operands for reuse,
to exchange data between lanes/waves of a workgroup, and to feed the
[matrix cores](mfma.md) at full rate. In HIP it is declared with `__shared__`;
the compiler lowers accesses to the `ds_*` instruction family.

LDS is organized as **32-bit-wide banks**. Successive 4-byte dwords map to
successive banks, and the banks can service independent accesses in parallel.
When two or more lanes in the same access cycle hit *different addresses in the
same bank*, the accesses **serialize** — a bank conflict, and the dominant LDS
performance pitfall.

## Capacity and bank geometry

| Property | gfx942 (CDNA3) | gfx950 (CDNA4) |
|---|---|---|
| LDS per CU | **64 KB** | **160 KB** |
| Banks | **32** | **64** |
| Bank width | 32 bits (1 dword) | 32 bits (1 dword) |
| Depth per bank | 512 dwords | 640 dwords |
| Allocation granularity | 512-byte blocks, 512-byte aligned | 512-byte blocks, 512-byte aligned |

CDNA4 both **doubles the bank count** (32 → 64) and grows capacity 2.5×. The
extra banks raise the conflict-free access width: a single wave64 `ds_read_b32`
can in principle touch 64 distinct banks instead of 32, so layouts that conflict
on gfx942 may run clean on gfx950 (and vice-versa — padding tuned for 32 banks
can be sub-optimal at 64). Always re-tune LDS strides when porting; see the
[gfx942 → gfx950 migration notes](../migration/gfx942-to-gfx950.md).

> Capacity also bounds occupancy. With 64 KB/CU on gfx942, a kernel using 32 KB
> of `__shared__` per workgroup caps the CU at 2 resident workgroups regardless
> of VGPR headroom. LDS-per-workgroup is therefore a first-class occupancy knob —
> see [occupancy tuning](../technique/occupancy-tuning.md).

## Bank conflicts in detail

A wave64 LDS instruction is **dispatched over the LDS in groups of 16 lanes per
cycle** (4 cycles for a full 64-lane wave). Within each 16-lane group the
hardware resolves bank addresses:

- **Conflict-free:** the active lanes map to distinct banks (or all read the same
  dword — a *broadcast*, which is also free).
- **N-way conflict:** N lanes target different addresses in one bank → the
  hardware replays that bank N times, stretching the op by roughly N× (observed
  serialization ranges from ~2 to 64 cycles for a fully conflicted wave).

The bank index for a byte address is `(address / 4) % num_banks`. The classic
example is staging a 32×32 FP32 tile and reading down a column: with a row stride
of 32 dwords, every element of a column lands in the same bank on gfx942
(`32 % 32 == 0`) → a 32-way conflict. The standard fix is to **pad the leading
dimension** so consecutive rows fall into different banks:

```cpp
// BAD: column reads all collide in one bank on gfx942 (stride 32 ≡ 0 mod 32)
__shared__ float tileA[32][32];

// GOOD: +1 dword pad breaks the stride alignment -> conflict-free columns
__shared__ float tileA[32][32 + 1];   // 33-dword stride: 33 % 32 == 1
```

Padding costs a little capacity; an alternative that costs none is a swizzled
(XOR-permuted) index — see [LDS swizzling](../technique/lds-swizzling.md) and the
[bank-conflict pattern](../patterns/bank-conflicts.md).

## The `ds_*` instruction family

LDS is accessed through dedicated Data Share instructions, *not* through
`buffer`/`global` ops. The common ones:

| Instruction | Width | Notes |
|---|---|---|
| `ds_read_b32` / `b64` / `b128` | 1 / 2 / 4 dwords | Wider reads amortize issue and improve bank utilization |
| `ds_write_b32` / `b64` / `b128` | 1 / 2 / 4 dwords | Same widths for stores |
| `ds_read2_b32` / `b64` | two strided elements | One instruction loads a pair at `base + offset0*stride` and `base + offset1*stride` |
| `ds_swizzle_b32` | 1 dword | Cross-lane dword shuffle within 32-lane groups; **no LDS storage used** |
| `ds_permute_b32` / `ds_bpermute_b32` | 1 dword | Arbitrary 64-lane gather (`permute`) / scatter (`bpermute`) via the LDS crossbar; **no LDS storage used** |

Two efficiency rules of thumb:

1. **Vectorize.** Prefer `ds_read_b128` over four `ds_read_b32`s — fewer issue
   slots and one address calculation. The compiler emits wide `ds_*` when your
   `__shared__` accesses are contiguous and aligned (e.g. via `float4`).
2. **`ds_read2_*` for strided pairs.** Double-buffered GEMM inner loops use it to
   pull two operand dwords per instruction at independent offsets.

`ds_swizzle`, `ds_permute`, and `ds_bpermute` are special: they reuse the LDS
crossbar for **lane-to-lane data movement without occupying LDS storage**. For
`ds_bpermute` the byte address is `lane_id * 4`, i.e. the *source lane index ×4*;
on an address collision the highest source lane wins; an out-of-range lane reads
0. These are the building blocks of cross-row wave reductions — see
[cross-lane operations](cross-lane.md) and [wave reduce](../technique/wave-reduce.md).

```cpp
// Cross-lane broadcast/shuffle with zero LDS storage cost.
// __builtin_amdgcn_ds_bpermute takes a BYTE address = srcLane * 4.
__device__ float shuffle_from(float v, int srcLane) {
    int addr = (srcLane & 63) << 2;                 // byte address
    int raw  = __builtin_amdgcn_ds_bpermute(addr, __builtin_bit_cast(int, v));
    return __builtin_bit_cast(float, raw);
}
```

## Allocation rules and the M0 register

LDS is carved out in **contiguous 512-byte blocks, 512-byte aligned**. The
runtime/compiler computes the per-workgroup LDS size from your `__shared__`
declarations (plus any dynamic `extern __shared__` request passed at launch) and
rounds it up to the allocation granularity.

A subtlety that bites hand-written assembly: many `ds_*` instructions clamp or
size their addressing against the **`M0` scalar register**, and `ds_*` offsets are
taken relative to the wave's LDS base. `M0` **must be initialized** before the
first dependent `ds_*` op (typically to the LDS size / `0xFFFFFFFF` for full
range). HIP/Clang handles this automatically; if you drop to inline GCN assembly
you are responsible for setting `M0` yourself.

```asm
; Minimal hand-written LDS round-trip (CDNA). M0 must be set for ds_* range.
    s_mov_b32      m0, 0xFFFFFFFF        ; enable full LDS addressing range
    v_lshlrev_b32  v1, 2, v0            ; v1 = tid * 4  (byte offset)
    ds_write_b32   v1, v2               ; LDS[tid] = v2
    s_waitcnt      lgkmcnt(0)           ; LDS ops gate on LGKMCNT, not VMCNT
    ds_read_b32    v3, v1               ; v3 = LDS[tid]
    s_waitcnt      lgkmcnt(0)
```

Note the **completion counter**: `ds_*` instructions retire against
**`LGKMCNT`** (the L/G/K/M counter), *not* `VMCNT`. A `__syncthreads()` lowers to
`s_barrier`, but a barrier alone does not guarantee your own prior `ds_write` is
visible to your own subsequent `ds_read` — pair barriers with the right
`s_waitcnt lgkmcnt(...)` semantics. See [s_waitcnt](s-waitcnt.md).

## Feeding the matrix cores

In a tiled GEMM the LDS sits between HBM and the [MFMA](mfma.md) units: global
tiles are staged into LDS, then `ds_read_b128` distributes operands into the VGPR
layout each `v_mfma_*` expects. Two techniques make this efficient:

- **Double buffering:** ping-pong two LDS tiles so `ds_read`/MFMA on tile *i*
  overlaps the global load of tile *i+1*
  ([lds-double-buffering](../technique/lds-double-buffering.md)).
- **Direct-to-LDS:** on CDNA the `buffer_load ... lds` / `global_load_lds_*`
  path streams HBM → LDS **bypassing VGPRs**, freeing registers for the
  accumulator tile ([async copy to LDS](async-copy-lds.md)).

AMD's step-by-step GEMM optimization blog walks through exactly these LDS
staging, vectorization, and bank-conflict-avoidance steps and their measured
impact on achieved TFLOPS.

## See also

- [Bank-conflict avoidance technique](../technique/bank-conflict-avoidance.md)
- [LDS swizzling](../technique/lds-swizzling.md)
- [Direct-to-LDS async copy](async-copy-lds.md)
- [Cross-lane operations (DPP / swizzle / permute)](cross-lane.md)
- [Bank-conflicts pattern](../patterns/bank-conflicts.md)

## Sources

- [AMD Instinct MI300 / CDNA3 ISA Reference Guide](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf)
- [AMD Instinct CDNA4 ISA Reference Guide](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-cdna4-instruction-set-architecture.pdf)
- [Optimizing a GEMM kernel on AMD Instinct (ROCm Blogs)](https://rocm.blogs.amd.com/artificial-intelligence/matrix-cores/README.html)
- [LLVM AMDGPU Backend User Guide](https://llvm.org/docs/AMDGPUUsage.html)
