---
id: pattern-bank-conflicts
title: LDS Bank Conflicts — serialized shared-memory access
type: pattern
architectures:
- gfx942
- gfx950
tags:
- bank-conflicts
- lds-bound
- serialized-lds
- lds
- ds-instructions
- swizzle
symptoms:
- bank-conflicts
- lds-bound
- serialized-lds
candidate_techniques:
- technique-lds-swizzling
- technique-bank-conflict-avoidance
related:
- hw-lds
- technique-lds-swizzling
- technique-bank-conflict-avoidance
- kernel-transpose-lds
- technique-vectorized-loads
sources:
- hw-lds
- doc-cdna3-isa
- doc-cdna4-isa
- blog-gemm-optimization
- ref-matrix-calculator
implemented_by:
- pr-FlyDSL-264
---
# LDS Bank Conflicts — serialized shared-memory access

## What this pattern looks like

A kernel that stages data through the [Local Data Share](../hardware/lds.md)
(LDS / `__shared__`) is spending far more time in `ds_read_*` / `ds_write_*`
than the byte volume justifies. Typical tells:

- The profiler attributes a large fraction of stall cycles to **LDS** rather
  than to VMEM or the matrix unit, even though the working set fits comfortably
  in 64 kB (gfx942) / 160 kB (gfx950).
- A GEMM or convolution that *should* be MFMA-bound instead reports the matrix
  core idling while waiting on `lgkmcnt` (the LDS/scalar counter — see
  [`s_waitcnt`](../hardware/s-waitcnt.md)).
- Replacing the LDS staging buffer with a deliberately padded layout makes the
  kernel measurably faster with **no change in arithmetic**.

These are all faces of one root cause: **two or more lanes of the same memory
cycle address the same LDS bank**, so the hardware *serializes* the access.

## Why it happens

On CDNA the LDS is split into **32 banks of 512 dwords** on gfx942 and **64
banks of 640 dwords** on gfx950; each bank is 32 bits (one dword) wide. A dword
at byte address `A` lives in bank:

```
bank = (A / 4) % num_banks      // num_banks = 32 (gfx942) or 64 (gfx950)
```

A wave64 `ds_read`/`ds_write` is dispatched to the LDS over **4 cycles, 16
lanes per cycle**. Within a single 16-lane group, if `k` lanes target distinct
dwords that map to the **same bank**, that group's access is split into `k`
serialized passes — a *k-way bank conflict*. The penalty ranges from a couple
of extra cycles up to ~64 cycles for a fully-degenerate 32-way pattern. An
access where every lane hits a different bank (or where lanes read the *same*
dword, which the hardware broadcasts for free) runs at full rate.

The classic trigger is a **power-of-two stride** equal to (a multiple of) the
bank count. Two examples that bite constantly:

```cpp
// gfx942: 32 banks. A 32-wide float column walk -> every row maps to bank 0.
__shared__ float tile[32][32];
float x = tile[threadIdx.x][0];   // lanes 0..31 all read column 0:
                                  // addresses 0, 128, 256, ... bytes
                                  // bank = (A/4) % 32 = 0 for ALL of them
                                  // => 32-way conflict, ~32x slower
```

Here the row stride is `32 floats = 128 bytes = 32 dwords`, an exact multiple
of the 32-bank width, so a column access collapses onto one bank. The same code
on gfx950 (64 banks) collapses 32 lanes onto bank 0 across two banks' worth of
addressing — still badly conflicted.

## Diagnosing it

1. **Profile.** With `rocprofv3` / Omniperf, look at the LDS section: a high
   *bank-conflict* count or a low *LDS access efficiency* / high LDS latency
   relative to issued `ds_*` instructions is the signal. Conflicts also inflate
   `lgkmcnt` wait time, which surfaces as matrix/VALU idle.
2. **Reason about the stride.** Write down the byte address each lane in a
   16-lane group produces, divide by 4, take `% num_banks`. If the set of
   resulting banks has duplicates, you have a conflict. Column-major access of a
   `[N][32]` (gfx942) or `[N][64]` (gfx950) tile is the canonical offender.
3. **A/B test padding.** Add one dword of column padding (below) and re-measure.
   A large delta confirms LDS serialization was the bottleneck rather than
   occupancy or VMEM.

## Fixes

### 1. Padding (cheapest, costs a little LDS)

Make the leading dimension **coprime with the bank count** by padding the inner
dimension by one (or a few) dwords. This shifts each successive row into a
different bank so a column walk fans out across all banks. See
[bank-conflict avoidance](../techniques/bank-conflict-avoidance.md).

```cpp
// gfx942: pad the 32-wide tile to 33 so the column stride is coprime with 32.
__shared__ float tile[32][33];     // +128 bytes total LDS
float x = tile[threadIdx.x][0];    // addresses 0, 132, 264, ...
                                   // bank = (A/4) % 32 = 0, 33%32=1, 66%32=2 ...
                                   // => all 32 distinct banks, conflict-free
```

Padding is the right tool when the conflicting stride is fixed and you can spare
a few hundred bytes of LDS without hurting occupancy.

### 2. Swizzling (no extra LDS, best for matrix tiles)

Permute the *storage* index with an XOR so that the bank a (row, col) maps to
varies with the row, eliminating conflicts for both the store and the later
column read **without** spending extra LDS — important when LDS capacity caps
occupancy. This is the [LDS swizzling](../techniques/lds-swizzling.md) technique
that CK and Tensile emit for GEMM LDS tiles:

```cpp
// XOR-swizzle a row-major [TILE][TILE] float tile (TILE a power of two).
// The xor spreads each logical column across all banks as the row advances.
__device__ inline int swz(int row, int col, int tile) {
    return row * tile + (col ^ (row & (tile - 1)));
}

__shared__ float smem[TILE * TILE];
smem[swz(r, c, TILE)] = g;                 // store
float v = smem[swz(threadIdx.x, c, TILE)]; // conflict-free column read
```

A hardware shortcut for the *read* side is `ds_swizzle_b32`, which permutes
dwords within 32-lane groups **without any LDS traffic at all** — useful for
small fixed transposes and reductions where you can avoid the round-trip
entirely. For arbitrary 64-lane gathers, `ds_bpermute_b32` routes through the
LDS crossbar without consuming LDS storage.

### 3. Restructure the access pattern

- Prefer **vectorized** `ds_read_b128` / `ds_write_b128` over four `b32` ops:
  fewer, wider transactions touch consecutive dwords and naturally spread across
  banks. See [vectorized loads](../techniques/vectorized-loads.md).
- For transposes, stage through LDS with a swizzled or padded layout so neither
  the write nor the read conflicts — the
  [LDS-staged transpose kernel](../kernels/transpose-lds.md) is a worked example.
- Lay tiles out so the *consuming* instruction (often an MFMA fragment load)
  reads consecutive dwords per lane. The
  [Matrix Instruction Calculator](../../sources/refs/ref-matrix-calculator.md)
  tells you exactly which register each lane needs, which lets you pick an LDS
  layout that feeds those reads conflict-free.

## Caveats

- **Broadcast is free, not a conflict.** If every lane in the 16-lane group
  reads the *identical* dword, the LDS broadcasts it in one pass. Only accesses
  to *different* dwords in the *same* bank serialize.
- **Bank count differs across architectures.** A layout tuned for 32 banks
  (gfx942) is not automatically conflict-free on 64 banks (gfx950), and vice
  versa. Parameterize the pad/swizzle by `num_banks` and re-profile when
  porting — see [gfx942 → gfx950 migration](../migration/gfx942-to-gfx950.md).
- **Padding trades LDS for bandwidth.** If your tile already pins occupancy via
  LDS capacity, prefer XOR swizzling (zero extra bytes) over padding.
- Conflicts only matter when LDS is actually on the critical path. Confirm with
  a profile before reshaping a layout; a kernel that is HBM-bound won't speed up
  by de-conflicting LDS.

## Sources

- [CDNA3 ISA Reference Guide — LDS / DS instructions](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf)
- [CDNA4 ISA Reference Guide](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-cdna4-instruction-set-architecture.pdf)
- [Optimizing GEMM on AMD GPUs (LDS staging & swizzling)](https://rocm.blogs.amd.com/artificial-intelligence/matrix-cores/README.html)
- [AMD Matrix Instruction Calculator](https://github.com/ROCm/amd_matrix_instruction_calculator)
