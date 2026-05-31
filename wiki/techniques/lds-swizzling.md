---
id: technique-lds-swizzling
title: LDS Swizzling — Conflict-Free A/B Tile Staging for MFMA
type: technique
architectures:
- gfx942
- gfx950
tags:
- lds-swizzling
- swizzled-layout
- bank-conflict-avoidance
- lds
- ds-instructions
- mfma
- swizzle
confidence: source-reported
reproducibility: snippet
hardware_features:
- lds
- ds-instructions
- mfma
- agpr
kernel_types:
- gemm
- hgemm
- fp8-gemm
languages:
- hip
- cpp
related:
- hw-lds
- technique-bank-conflict-avoidance
- technique-lds-double-buffering
- technique-vectorized-loads
- hw-mfma
- pattern-bank-conflicts
sources:
- hw-lds
- technique-bank-conflict-avoidance
- blog-gemm-optimization
- doc-cdna3-isa
- blog-matrix-cores-cdna
- ref-composable-kernel
implemented_by:
- pr-Tensile-1383
- pr-composable_kernel-2096
- pr-composable_kernel-1838
- pr-Tensile-1521
- pr-Tensile-1529
- pr-Tensile-1873
- pr-Tensile-1656
- pr-Tensile-1519
---
# LDS Swizzling — Conflict-Free A/B Tile Staging for MFMA

## Overview

In a tiled GEMM the A and B operand tiles are streamed HBM → LDS once per K-step
and then read many times by the matrix core. If the LDS layout is the naive
row-major `tile[row][col]`, the access pattern that feeds
[`v_mfma_*`](../hardware/mfma.md) frequently lands many lanes on the **same LDS
bank in the same cycle**, serializing the read. **LDS swizzling** permutes the
column index of each row with a cheap XOR so that the lanes participating in one
`ds_read` spread across distinct banks — eliminating the conflict without
changing the logical tile contents, adding any LDS traffic, or touching the
global-memory load pattern.

This is the structural cousin of [generic bank-conflict
avoidance](bank-conflict-avoidance.md) (which also offers padding): swizzling is
preferred for MFMA staging because, unlike padding, it wastes **zero** LDS
capacity — important when you are already double-buffering and LDS-capacity
bound.

## Why the naive layout conflicts

On gfx942 LDS has **32 banks of 32-bit words**; on gfx950, **64 banks**
(see [the LDS page](../hardware/lds.md)). A bank conflict is two or more lanes
addressing different words in the *same* bank in the same cycle; the hardware
serializes them (2–64 cycles). A wave64 `ds_read` is dispatched over 4 cycles,
**16 lanes per cycle**, so the question is whether each 16-lane group hits 16
distinct banks.

Consider an FP16 A-tile stored row-major as `__half a_lds[64][16]` and an MFMA
layout where 16 consecutive lanes read down a *column* (the same `col`,
successive `row`). The byte address of `a_lds[row][col]` is
`(row*16 + col)*2`, so its bank is `((row*16 + col)*2 / 4) % 32 = (row*8 + col) % 32`.
For a fixed `col`, stepping `row` by 1 steps the bank by 8 — so rows
`0,4,8,12,...` all map to the **same bank**. Sixteen lanes collapse onto 4
banks: a 4-way conflict on every operand read.

## The XOR swizzle

Replace the stored column with `col ^ swizzle(row)`, where `swizzle` injects the
high row bits into the column index so equal-`col` accesses fan out across banks.
A general form that works for a `ds_read_b128` (8-element FP16 vector) tile is:

```cpp
// One K-slab of an A tile staged in LDS, swizzled to be MFMA-conflict-free.
// TILE_M rows x TILE_K cols of __half. Vector width = 8 halves (ds_read_b128).
constexpr int TILE_M = 64;
constexpr int TILE_K = 32;
constexpr int VEC    = 8;                 // 128-bit ds access granularity
constexpr int COLS_V = TILE_K / VEC;      // vectors per row (=4)

__device__ __forceinline__
int swizzled_offset(int row, int col_vec) {
    // XOR the low bits of the row into the vector-column index.
    // COLS_V is a power of two, so the permutation is a bijection on [0,COLS_V).
    int x = col_vec ^ (row & (COLS_V - 1));
    return row * COLS_V + x;              // index in units of VEC halves
}

__device__ void store_A_tile(__half2_8 *lds, const __half2_8 *g, int lane) {
    // 64 rows x 4 vec-cols = 256 vectors; wave64 writes 64 per pass.
    #pragma unroll
    for (int i = 0; i < TILE_M * COLS_V / 64; ++i) {
        int idx = i * 64 + lane;
        int row = idx / COLS_V, col_vec = idx % COLS_V;
        lds[swizzled_offset(row, col_vec)] = g[idx];   // ds_write_b128
    }
}

__device__ __half2_8 load_A_for_mfma(const __half2_8 *lds, int row, int col_vec) {
    return lds[swizzled_offset(row, col_vec)];         // ds_read_b128
}
```

The key property: **store and load use the identical `swizzled_offset`**, so the
permutation cancels — `load_A_for_mfma(row, col_vec)` returns exactly the
logical element that `store_A_tile` wrote for `(row, col_vec)`. Correctness is
preserved while the physical bank assignment changes.

After swizzling, the bank for the 16 lanes that previously collided becomes
`((row*COLS_V + (col_vec ^ (row & (COLS_V-1)))) ...) % banks`, and the
row-dependent XOR term breaks the arithmetic progression that caused the
collapse — the 16 lanes now cover 16 distinct banks. Verify with the profiler
rather than by hand (see below).

## Choosing the swizzle width

The XOR mask must (1) be a power-of-two minus one so the map is a bijection over
the vector-columns, and (2) inject enough row bits to cover the number of banks
touched per cycle. Rules of thumb:

- Swizzle at the **vector granularity you actually `ds_read`** (`b32`/`b64`/`b128`),
  not per scalar element — the bank stride is set by the access width.
- Pick the mask so that `log2(banks_per_cycle)` row bits participate. On gfx942
  the 16 lanes/cycle want 16 distinct banks; on gfx950 the same pattern has
  twice the banks (64) to play with, so a too-narrow gfx942 mask is harmless but
  may leave a wider tile sub-optimal — re-tune per arch.
- Keep the swizzle consistent across **both** double-buffer halves so the steady
  state never reverts to a conflicting layout — see
  [LDS double-buffering](lds-double-buffering.md).

## Verifying with the profiler

Swizzling is easy to get subtly wrong; always confirm against hardware counters
instead of trusting the algebra.

```bash
# Count LDS bank conflicts for the kernel of interest.
rocprofv3 --pmc SQ_LDS_BANK_CONFLICT SQ_LDS_IDX_ACTIVE -- ./gemm_bench
# Goal: SQ_LDS_BANK_CONFLICT / SQ_LDS_IDX_ACTIVE  -> ~0 after swizzling.
```

A non-zero conflict ratio that drops to near-zero after applying the XOR (with
identical kernel timing math otherwise) is the signal the swizzle is effective.
In a tiled HGEMM that is otherwise LDS-read bound, removing a 4-way conflict on
operand loads recovers the cycles the matrix core spent stalled on LDS — see the
end-to-end walkthrough in the
[ROCm GEMM optimization blog](https://rocm.blogs.amd.com).

## Relation to library practice

You rarely write this by hand in production: [Composable
Kernel](../languages/composable-kernel.md) describes operand tiles with
*distributed-tensor* layout transforms whose LDS descriptors already encode an
XOR swizzle, and hipBLASLt/Tensile bake equivalent `MatrixInstruction`-aware LDS
layouts into their assembly. Hand-rolling the swizzle is worth it when you write
a custom fused kernel (e.g. attention or MoE) and the autogenerated path does not
cover your tile shape.

## Pitfalls

- **Forgetting symmetry.** Store and load must use the *same* offset function.
  An asymmetric swizzle silently corrupts results.
- **Swizzling the wrong axis.** Swizzle the dimension that varies *across lanes
  within a cycle*, which depends on the MFMA register layout — derive it with the
  [Matrix Instruction Calculator](../../sources/refs/ref-matrix-calculator.md),
  not by guessing.
- **Mismatched access width.** If you `ds_write_b128` but `ds_read_b64`, the bank
  stride differs between store and load; design the swizzle for the read width.
- **Assuming gfx942 masks transfer to gfx950.** Bank count doubled (32 → 64);
  re-profile after porting.

## See also

- [LDS hardware: banks & conflicts](../hardware/lds.md)
- [Bank-conflict avoidance (padding vs swizzle)](bank-conflict-avoidance.md)
- [LDS double-buffering](lds-double-buffering.md)
- [MFMA matrix instructions](../hardware/mfma.md)
- [Pattern: bank conflicts](../patterns/bank-conflicts.md)

## Sources

- [CDNA3 ISA Reference Guide — LDS & ds_* instructions](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf)
- [Matrix Core Programming on CDNA3 and CDNA4](https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores-cdna/README.html)
- [Optimizing GEMM on AMD GPUs (ROCm blog)](https://rocm.blogs.amd.com)
- [Composable Kernel — distributed tensor LDS layouts](https://github.com/ROCm/composable_kernel)
