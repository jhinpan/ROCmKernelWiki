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
version_sensitive:
- vs-lds-phase-groups-gfx942-gfx950
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
serializes them. The relevant competing lanes come from the emitted opcode's
architecture-specific **phase group**. For example, `ds_read_b128` uses eight
eight-lane, non-contiguous groups on gfx942 and four 16-lane groups on gfx950;
see the complete [LDS phase tables](../hardware/lds.md).

Consider a row-major FP16 tile `a_lds[64][32]`, viewed as four 128-bit vectors
per row, and the simplified mapping where lane `i` reads row `i` at one fixed
vector-column `col_vec`. The vector's starting bank is
`4 * (4*row + col_vec) mod num_banks`, because every vector occupies four
dwords. Equivalently, measure 4-bank vector slots: `(4*row + col_vec) mod 16`
on gfx950 or `mod 8` on gfx942.

Enumerating the actual b128 phase groups shows the conflict. Without a swizzle,
each gfx950 16-lane phase reaches only four start slots, each repeated four
times; each gfx942 eight-lane phase reaches two start slots, also repeated four
times. Thus this simplified row-major mapping has a 4-way conflict on every
dword of the vector access — not a generic “consecutive 16 lanes” rule.

## The XOR swizzle

Replace the stored vector-column with `col_vec ^ swizzle(row)`, where `swizzle`
injects selected row bits into the column index. The exact bits are a property
of the lane-to-row mapping, emitted opcode, and architecture — there is no one
universal XOR mask. For the simplified model above (**lane id equals `row`**,
every lane in a phase reads the same `col_vec`, four vector-columns per row),
the reported b128 phase tables give these target-specific masks:

```cpp
// Schematic 64x32 FP16 tile; offsets are in units of 8-half (128-bit) vectors.
constexpr int COLS_V = 4;

__device__ __forceinline__
int schematic_b128_offset(int row, int col_vec) {
#if !defined(__HIP_DEVICE_COMPILE__)
    int x = col_vec;  // host parse pass; this __device__ helper is not executed
#elif defined(__gfx950__)
    int x = col_vec ^ ((row >> 2) & 3);
#elif defined(__gfx942__)
    int x = col_vec ^ ((row >> 1) & 3);
#else
#error "derive the b128 swizzle for this target's phase groups"
#endif
    return row * COLS_V + x;
}
```

Use the identical function on the store and load paths so the permutation
cancels logically. In this specific model, divide each 128-bit start-bank index
by four: gfx950 evaluates `(4*row + x) mod 16`, which covers `0..15` once in
each reported 16-lane b128 phase; gfx942 evaluates `(4*row + x) mod 8`, which
covers `0..7` once in each reported eight-lane phase. Because each vector then
occupies four adjacent banks, the phase covers all 64 or 32 banks without
overlap. This proof does **not** transfer automatically to a real MFMA fragment
whose lane-to-`(row,col_vec)` mapping differs. Enumerate that actual mapping
against the phase table, then verify with the profiler (see below).

## Choosing the swizzle width

The XOR mask must (1) be a power-of-two minus one so the map is a bijection over
the vector-columns, and (2) inject enough row bits to cover the number of banks
touched per cycle. Rules of thumb:

- Swizzle at the **vector granularity you actually `ds_read`** (`b32`/`b64`/`b128`),
  not per scalar element — the bank stride is set by the access width.
- Pick the mask so lanes in each actual phase group map to distinct banks. The
  group membership changes with `b32`/`b64`/`b128` and between gfx942/gfx950;
  a swizzle proven for one combination is not automatically safe or optimal for
  another. Re-tune and measure per emitted opcode and architecture.
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
