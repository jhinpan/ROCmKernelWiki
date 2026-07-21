---
id: hw-lds
title: LDS — Local Data Share (CDNA3/CDNA4 Shared Memory)
type: hardware
version_sensitive:
- vs-lds-size-gfx950
- vs-lds-phase-groups-gfx942-gfx950
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
- blog-amdgpu-kernel-opt-guide
aliases:
- LDS
- local data share
- shared memory
- groupshared
- __shared__
implemented_by:
- pr-FlyDSL-191
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
| Allocation granularity | 512-byte units | 1280-byte units |

CDNA4 both **doubles the bank count** (32 → 64) and grows capacity 2.5×. The
extra banks raise the conflict-free access width: a single wave64 `ds_read_b32`
can in principle touch 64 distinct banks instead of 32, so layouts that conflict
on gfx942 may run clean on gfx950 (and vice-versa — padding tuned for 32 banks
can be sub-optimal at 64). Always re-tune LDS strides when porting; see the
[gfx942 → gfx950 migration notes](../migration/gfx942-to-gfx950.md).

> Capacity also bounds occupancy. With 64 KB/CU on gfx942, a kernel using 32 KB
> of `__shared__` per workgroup caps the CU at 2 resident workgroups regardless
> of VGPR headroom. LDS-per-workgroup is therefore a first-class occupancy knob —
> see [occupancy tuning](../techniques/occupancy-tuning.md).

## Bank conflicts in detail

There is no single “16 lanes/cycle for every LDS instruction” rule. The LDS
crossbar divides a wave into **instruction- and architecture-specific phase
groups**. Bank conflicts are arbitrated only among lanes in the same phase:

The captured guide separately reports an **address-send** limit on gfx942 of up
to 16 addresses per SIMD per cycle and up to 32 addresses per CU per cycle,
before the instruction-specific data phases below. That number is retained as a
guide-reported, not independently reproduced, front-end limit; it must not be
substituted for the phase groups of every `ds_*` opcode.

- **Conflict-free:** the active lanes map to distinct banks (or all read the same
  dword — a *broadcast*, which is also free).
- **N-way conflict:** N lanes target different addresses in one bank → the
  hardware waterfalls/replays that bank. Different phase groups already execute
  separately and do not conflict with one another.

Within one conflicting phase, the guide describes arbitration as selecting the
lowest-thread-id subset whose accesses do not conflict, then replaying the
leftover lanes. The worst case of distinct addresses all mapping to one bank
therefore becomes a one-lane-at-a-time waterfall. This arbitration detail is
guide-reported; same-address reads remain the broadcast exception.

### gfx942 / CDNA3 phase groups

| Instruction | Phase groups (wave64 lane ids) | Status |
|---|---|---|
| `ds_read_b32` | `0–31`; `32–63` | upstream empirical result |
| `ds_read_b64` | `0–15`; `16–31`; `32–47`; `48–63` | upstream empirical result |
| `ds_read2_b64` | the four b64 groups for offset 0, then the four for offset 1 (eight phases total) | guide-reported; not covered by the current read-phase harness |
| `ds_write_b128` | contiguous groups `0–7`, `8–15`, …, `56–63` | guide-reported; not covered by the current read-phase harness |

`ds_read_b128` has eight non-contiguous eight-lane phases:

| Phase | Lanes | Phase | Lanes |
|---:|---|---:|---|
| 0 | `0–3 + 20–23` | 4 | `8–11 + 28–31` |
| 1 | `32–35 + 52–55` | 5 | `40–43 + 60–63` |
| 2 | `4–7 + 16–19` | 6 | `12–15 + 24–27` |
| 3 | `36–39 + 48–51` | 7 | `44–47 + 56–59` |

### gfx950 / CDNA4 phase groups

| Instruction | Phase groups (wave64 lane ids) | Status |
|---|---|---|
| `ds_read_b32` | `0–63` | upstream empirical; reproduced on MI355X |
| `ds_read_b64` | `0–31`; `32–63` | upstream empirical; reproduced on MI355X |
| `ds_read_b128` | see below | upstream empirical; local automatic classifier was inconclusive |

The four reported gfx950 `ds_read_b128` phases are:

| Phase | Lanes |
|---:|---|
| 0 | `0–3, 12–15, 20–23, 24–27` |
| 1 | `32–35, 44–47, 52–55, 56–59` |
| 2 | `4–7, 8–11, 16–19, 28–31` |
| 3 | `36–39, 40–43, 48–51, 60–63` |

The ranges in the guide that end at `T64` are endpoint typos: wave64 lanes run
from `T0` through `T63`. The full upstream tables and harness are preserved in
the pinned
[`empirical-lds` summary](https://github.com/nod-ai/amd-shark-ai/blob/efa471aeef66a260c85983cc41e833bfa769dade/docs/empirical-lds/lds_summary.md).

> **2026-07-20 MI355X check.** On an idle gfx950/ROCm 7.1.1 device, the upstream
> harness classified the 64-bit stride sweep as **64 banks** and reproduced the
> one b32 group (`0–63`) and two b64 groups (`0–31`, `32–63`). A three-run b128
> automatic classification was unstable and did not reproduce the known groups,
> so this pass deliberately leaves the b128 table as upstream-empirical rather
> than claiming an independent verification. gfx942 runtime access was blocked
> by the unavailable VPN route, so its tables also remain upstream-empirical.

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
(XOR-permuted) index — see [LDS swizzling](../techniques/lds-swizzling.md) and the
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
| `ds_permute_b32` / `ds_bpermute_b32` | 1 dword | Arbitrary 64-lane push/scatter (`permute`) / pull/gather (`bpermute`) via the LDS crossbar; **no LDS storage used** |

Two efficiency rules of thumb:

1. **Vectorize.** Prefer `ds_read_b128` over four `ds_read_b32`s — fewer issue
   slots and one address calculation. The compiler emits wide `ds_*` when your
   `__shared__` accesses are contiguous and aligned (e.g. via `float4`).
2. **`ds_read2_*` for strided pairs.** Double-buffered GEMM inner loops use it to
   pull two operand dwords per instruction at independent offsets.

`ds_swizzle`, `ds_permute`, and `ds_bpermute` are special: they reuse the LDS
crossbar for **lane-to-lane data movement without occupying LDS storage**. For
the pull/gather `ds_bpermute`, each lane's byte address is the requested *source
lane index ×4* and an out-of-range source reads 0. For the push/scatter
`ds_permute`, each source lane names a destination; if several sources collide,
the highest source lane wins. These are the building blocks of cross-row wave
reductions — see
[cross-lane operations](cross-lane.md) and [wave reduce](../techniques/wave-reduce.md).

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

LDS is allocated per workgroup in architecture-specific units: **512 bytes on
gfx942** and **1280 bytes on gfx950**. The runtime/compiler computes the required
size from `__shared__` declarations (plus any dynamic `extern __shared__` request
passed at launch) and rounds it up to the target's unit. Occupancy calculations
must use that rounded value; for example, 32 KiB becomes 33,280 bytes on gfx950.

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
  ([lds-double-buffering](../techniques/lds-double-buffering.md)).
- **Direct-to-LDS:** on CDNA the `buffer_load ... lds` / `global_load_lds_*`
  path streams HBM → LDS **bypassing VGPRs**, freeing registers for the
  accumulator tile ([async copy to LDS](async-copy-lds.md)).

AMD's step-by-step GEMM optimization blog walks through exactly these LDS
staging, vectorization, and bank-conflict-avoidance steps and their measured
impact on achieved TFLOPS.

> **gfx950 bank index & the XOR-swizzle rule.** On MI350X/MI355X the LDS is
> **160 KB with 64 banks** (640 × 4 B entries, 256 B/clock read), so the bank a
> Dword lands in is `(address / 4) % 64` (vs `% 32` on MI300). For the column-wise
> `ds_read_b128` access an MFMA tile load performs, **padding is awkward to apply**
> — the shark-ai guide recommends **XOR-based swizzling instead of padding** to
> spread those 128-bit reads across all 64 banks. Wide LDS instructions
> (`ds_read_b128`, `ds_read2_b64`, `ds_write_b128`) and `ds_*` over `flat_*` are
> preferred for the same reason. See
> [LDS swizzling](../techniques/lds-swizzling.md).

## See also

- [Bank-conflict avoidance technique](../techniques/bank-conflict-avoidance.md)
- [LDS swizzling](../techniques/lds-swizzling.md)
- [Direct-to-LDS async copy](async-copy-lds.md)
- [Cross-lane operations (DPP / swizzle / permute)](cross-lane.md)
- [Bank-conflicts pattern](../patterns/bank-conflicts.md)

## Sources

- [AMD Instinct MI300 / CDNA3 ISA Reference Guide](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf)
- [AMD Instinct CDNA4 ISA Reference Guide](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-cdna4-instruction-set-architecture.pdf)
- [Optimizing a GEMM kernel on AMD Instinct (ROCm Blogs)](https://rocm.blogs.amd.com/artificial-intelligence/matrix-cores/README.html)
- [AMDGPU Kernel Optimization Guide (captured snapshot)](https://github.com/nod-ai/amd-shark-ai/blob/efa471aeef66a260c85983cc41e833bfa769dade/docs/amdgpu_kernel_optimization_guide.md) — gfx950 LDS specs, phase groups, and XOR-swizzle guidance
- [ROCm Blogs — Avoiding LDS bank conflicts on AMD GPUs](https://rocm.blogs.amd.com/software-tools-optimization/lds-bank-conflict/README.html)
- [LLVM AMDGPU Backend User Guide](https://llvm.org/docs/AMDGPUUsage.html)
