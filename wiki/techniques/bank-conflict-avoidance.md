---
id: technique-bank-conflict-avoidance
title: "LDS Bank-Conflict Avoidance (padding, swizzle, ds_read2)"
type: technique
architectures:
- gfx942
- gfx950
tags:
- bank-conflict-avoidance
- lds
- ds-instructions
- swizzle
- lds-swizzling
- swizzled-layout
confidence: source-reported
reproducibility: snippet
hardware_features:
- lds
- ds-instructions
- swizzle
kernel_types:
- gemm
- transpose
languages:
- hip
related:
- hw-lds
- technique-lds-swizzling
- technique-vectorized-loads
- technique-lds-double-buffering
- kernel-transpose-lds
- pattern-bank-conflicts
sources:
- hw-lds
- doc-cdna3-isa
- doc-cdna4-isa
- blog-gemm-optimization
- blog-cktile-gemm
---

# LDS Bank-Conflict Avoidance (padding, swizzle, ds_read2)

## Overview

The [Local Data Share (LDS)](../hardware/lds.md) is split into independent
**banks**, each 32 bits wide. A *bank conflict* occurs when more than one lane in
the same access cycle addresses **different dwords that map to the same bank** —
the hardware cannot satisfy them in parallel, so it serializes the accesses,
adding 2–64 cycles of latency. For an LDS-bound kernel (a GEMM mainloop reading
A/B tiles, a staged transpose) conflicts directly throttle the matrix core.

Bank count differs by architecture, so the conflict-avoidance stride differs:

| Arch | LDS / CU | Banks | Dwords/bank | Bank of dword `d` |
|---|---|---|---|---|
| gfx942 (CDNA3) | 64 kB | **32** | 512 | `d % 32` |
| gfx950 (CDNA4) | 160 kB | **64** | 640 | `d % 64` |

A wave64 LDS instruction is **dispatched over 4 cycles, 16 lanes per cycle**.
That means a conflict is evaluated within each 16-lane quarter-wave, not across
all 64 lanes at once — a useful subtlety when reasoning about which lanes collide.

## Where conflicts come from

The classic case is a column-major access into a row-major LDS tile. Store an
`N×N` FP32 tile contiguously and have each lane read down a column:

```
addr(row, col) = row * N + col      // dword index
bank(row, col) = (row * N + col) % 32   // gfx942
```

If `N` is a multiple of 32 (e.g. a 64×64 tile), every row of a fixed column lands
on the **same bank** → a 32-way conflict on gfx942. The fix is to break the
power-of-two stride.

## Fix 1 — padding the leading dimension

Pad each row by one (or a few) dwords so the per-row stride is coprime with the
bank count. For FP32 on gfx942, `LDS_STRIDE = N + 1` shifts every successive row
into a different bank:

```cpp
// gfx942: 32 banks. Pad column-major tile so a column walk hits 32 distinct banks.
#define TILE 64
#define PAD  1                       // +1 dword/row breaks the %32 aliasing
__shared__ float tileA[TILE][TILE + PAD];

__device__ void load_and_consume(const float* __restrict__ g, int ld) {
    int r = threadIdx.y, c = threadIdx.x;            // 0..63
    tileA[r][c] = g[r * ld + c];                      // coalesced global -> LDS
    __syncthreads();                                  // s_barrier

    // Column walk: lane t reads tileA[t][myCol]. With PAD=1 each successive
    // row advances the bank by (TILE+PAD)%32 = 1, so 32 lanes -> 32 banks.
    float v = tileA[threadIdx.x][myCol];
    // ... feed MFMA / accumulate ...
}
```

Cost: padding wastes a little LDS (here `64 * 1 * 4 B = 256 B` per tile) and can
lower occupancy if you are already LDS-bound. Verify the chosen pad with the LDS
budget — see [occupancy tuning](occupancy-tuning.md). On gfx950 the bank count is
64, so a `+1` pad still works, but a half-tile pad pattern that was conflict-free
on 32 banks may *not* be on 64 — always recompute `stride % banks`.

## Fix 2 — XOR swizzle (no wasted LDS)

Padding spends LDS capacity. A **swizzle** instead permutes the column index with
a XOR so that the same logical tile occupies a *dense* footprint while still
spreading banks. The standard pattern XORs the column with a field of the row:

```cpp
// Conflict-free, zero padding. Map logical (row,col) -> physical column.
__device__ inline int swizzle(int row, int col) {
    // XOR high bits of row into col; keeps it in-range for power-of-two TILE.
    return col ^ (row & (TILE - 1));      // tune mask to vector width
}

__shared__ float tileA[TILE][TILE];       // dense, no PAD
// store
tileA[r][swizzle(r, c)] = g[r * ld + c];
__syncthreads();
// load (column walk) — physical columns now span distinct banks
float v = tileA[threadIdx.x][swizzle(threadIdx.x, myCol)];
```

The swizzle must be applied **identically on store and load** so the data lands
where the reader looks. Match the XOR granularity to the access width: a
`ds_read_b128` moves 4 dwords/lane, so swizzle on 4-dword (16-byte) granules, not
single dwords, or you reintroduce conflicts within the vector. This is the
mechanism Composable Kernel and Tensile use for their A/B LDS layouts; see the
dedicated [LDS swizzling](lds-swizzling.md) page for the full derivation.

## Fix 3 — vectorize and use `ds_read2`

Wider LDS instructions move more data per issue *and* change how addresses map to
banks. Prefer `ds_read_b128` (and aligned `float4`/`__shared__` accesses) so each
lane pulls a contiguous 16-byte run that naturally spans 4 consecutive banks.

`ds_read2_b32` / `ds_read2_b64` load **two strided elements per lane in one
instruction** (two independent offsets `offset0`, `offset1`). They are ideal for
double-buffered or strided tile reads because the two halves are scheduled
together and you halve instruction issue:

```asm
; Two FP32 dwords per lane, offsets in units of the element size.
; offset1-offset0 chosen so the two reads land on different banks.
ds_read2_b32  v[2:3], v0 offset0:0 offset1:8
s_waitcnt     lgkmcnt(0)            ; LDS reads gate on LGKMCNT
```

The compiler emits `ds_read2_*` automatically when two `__shared__` loads share a
base and have a constant, in-range stride — keep the offsets within the 8-bit
scaled `offset` field (≤ 255 elements apart) so the fusion fires.

## Verifying with the profiler

Measure, do not guess. `rocprof` / `rocprofiler-compute` expose LDS bank-conflict
counters:

```bash
rocprof --stats \
  --metric SQ_LDS_BANK_CONFLICT SQ_LDS_IDX_ACTIVE \
  ./my_kernel
# bank-conflict cycles / total LDS cycles -> target near 0%
```

A high `SQ_LDS_BANK_CONFLICT` ratio is the defining symptom of the
[bank-conflicts pattern](../patterns/bank-conflicts.md). Iterate: tweak pad or
swizzle mask, re-measure, and confirm the MFMA mainloop is no longer
LDS-issue-bound.

## Pitfalls

- **Recomputing for gfx950.** 32→64 banks changes which strides alias. A layout
  hand-tuned for MI300X may regress on MI350X; recompute `stride % banks`.
- **Vector-width mismatch.** Swizzling per-dword while reading `b128` leaves
  intra-vector conflicts. Swizzle at the access granularity.
- **Over-padding.** Padding that pushes the tile past an LDS occupancy cliff can
  cost more than the conflicts it removes — check waves/CU.
- **Store side forgotten.** A swizzle applied only on read scrambles data; apply
  the same permutation on write.

## See also

- [LDS hardware reference](../hardware/lds.md)
- [LDS swizzling](lds-swizzling.md)
- [Vectorized LDS/global loads](vectorized-loads.md)
- [Bank-conflict pattern](../patterns/bank-conflicts.md)
- [LDS-staged transpose kernel](../kernels/transpose-lds.md)

## Sources

- [AMD CDNA3 ISA Reference Guide — LDS / `ds_read*`, `ds_read2_*`](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf)
- [AMD CDNA4 ISA Reference Guide — 64-bank LDS](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-cdna4-instruction-set-architecture.pdf)
- [Optimizing GEMM on AMD GPUs (ROCm Blogs)](https://rocm.blogs.amd.com/artificial-intelligence/matrix-cores/README.html)
- [CK-Tile GEMM walkthrough (ROCm Blogs)](https://rocm.blogs.amd.com/software-tools-optimization/ck-tile-gemm/README.html)
