---
id: kernel-transpose-lds
title: LDS-Staged Bank-Conflict-Free Matrix Transpose
type: kernel
architectures:
- gfx942
- gfx950
tags:
- transpose
- lds
- ds-instructions
- bank-conflict-avoidance
- lds-swizzling
- swizzled-layout
- vectorized-loads
confidence: inferred
reproducibility: runnable
version_sensitive:
- vs-lds-phase-groups-gfx942-gfx950
artifact_dir: examples/transpose-lds
kernel_types:
- transpose
- elementwise
languages:
- gcn-asm
- hip
related:
- hw-lds
- technique-lds-swizzling
- technique-bank-conflict-avoidance
- pattern-bank-conflicts
- kernel-bandwidth-microbench
sources:
- ref-gcnasm
- hw-lds
- technique-lds-swizzling
- doc-cdna3-isa
- doc-mi300x-datasheet
- doc-llvm-amdgpu
performance_claims:
- gpu: MI300X
  dtype: fp32
  metric: effective bandwidth (read+write)
  value: ~4.2 TB/s (~80% of 5.3 TB/s HBM3 peak)
  utilization: ~0.80 of HBM peak
  baseline: naive strided transpose (uncoalesced stores)
  source_id: doc-mi300x-datasheet
- gpu: MI300X
  dtype: fp32
  metric: LDS bank conflicts per transposed tile read
  value: 0 (padded/swizzled) vs 32-way (unpadded)
  source_id: doc-cdna3-isa
implemented_by:
- pr-composable_kernel-3027
- pr-composable_kernel-3592
- pr-composable_kernel-3038
- pr-composable_kernel-3465
- pr-composable_kernel-94
- pr-composable_kernel-2955
- pr-composable_kernel-2905
- pr-composable_kernel-984
---
# LDS-Staged Bank-Conflict-Free Matrix Transpose

## Overview

An out-of-place transpose `out[x][y] = in[y][x]` is **memory-bound**: it moves
each element exactly once in and once out, with zero arithmetic. The only thing
that matters is whether both the global read and the global write are
**coalesced**. A naive kernel that reads `in` row-major and writes `out`
directly cannot be coalesced on both ends — one side always becomes a strided
(stride = `rows` or `cols`) access pattern, which serializes into many partial
cache-line transactions and collapses effective bandwidth.

The standard fix (the `transpose-lds` pattern in
[gcnasm](../../sources/refs/ref-gcnasm.md)) is to **stage a square tile through
LDS**: load a `TILE × TILE` block with a coalesced row-major read, transpose it
*inside* LDS, then store it with a coalesced row-major write. The transpose
itself becomes a column-wise LDS read — which is exactly where **bank conflicts**
appear and must be designed away.

## Why the LDS read conflicts

On gfx942 the LDS has **32 banks of 512 dwords, each 32-bit wide**; on gfx950 it
is **64 banks of 640 dwords** (see [LDS](../hardware/lds.md)). The bank of a
dword address is `(addr/4) mod num_banks`. A wave64 access is divided into
architecture- and opcode-specific phase groups. A scalar `ds_read_b32`
uses two 32-lane phases on gfx942 and one 64-lane phase on gfx950; b64/b128 use
smaller, different groups. Two lanes in the same phase that hit different
addresses in one bank serialize. Inspect the emitted width and consult the
[LDS phase tables](../hardware/lds.md) before declaring a layout conflict-free.

Store the tile naively as `float tile[TILE][TILE]` with `TILE = 32`. The
transposed read `tile[col][row]` walks **down a column**: for a fixed column,
successive rows are `TILE = 32` dwords apart, so every address maps to
`(row*32) mod 32 = 0` — **the same bank**. That is a 32-way conflict on every
transposed read, the worst case.

## Fix 1 — architecture-aware padding

On gfx942, pad the row stride to `TILE + 1 = 33` dwords. Each b32 phase sees
one column across 32 rows, and `gcd(33, 32) = 1`, so those rows land in 32
distinct banks. On gfx950 a b32 phase is all 64 lanes: with `block(32,32)` it
contains two adjacent columns across the same 32 rows. Stride 33 makes those two
bank sets overlap; `TILE + 2 = 34` instead maps one column to the 32 even banks
and the other to the 32 odd banks. Thus this particular thread mapping needs a
target-specific pad.

```cpp
// MI300X / MI350X — out-of-place fp32 transpose, LDS-staged.
// Launch: dim3 block(TILE, TILE); dim3 grid((cols+TILE-1)/TILE,(rows+TILE-1)/TILE);
#define TILE 32
#if defined(__gfx950__)
#define LDS_PAD 2  // wave64 phase contains two columns: split even/odd banks
#else
#define LDS_PAD 1  // gfx942 b32 phases
#endif
__global__ void transpose_lds(float* __restrict__ out,
                              const float* __restrict__ in,
                              int rows, int cols) {
    __shared__ float tile[TILE][TILE + LDS_PAD];

    int x = blockIdx.x * TILE + threadIdx.x;   // input column
    int y = blockIdx.y * TILE + threadIdx.y;   // input row

    // Coalesced read: consecutive threadIdx.x -> consecutive global columns.
    if (x < cols && y < rows)
        tile[threadIdx.y][threadIdx.x] = in[(size_t)y * cols + x];

    __syncthreads();                            // -> s_barrier

    // Output tile is the transposed block: swap block coordinates.
    int xo = blockIdx.y * TILE + threadIdx.x;   // output column (= input row)
    int yo = blockIdx.x * TILE + threadIdx.y;   // output row    (= input col)

    // Transposed LDS read (down a column) + coalesced global write.
    if (xo < rows && yo < cols)
        out[(size_t)yo * rows + xo] = tile[threadIdx.x][threadIdx.y];
}
```

`block(32,32)` = 1024 work-items = **16 wave64s** per workgroup. The padded tile
costs 4224 B on gfx942 or 4352 B on gfx950, leaving ample per-CU LDS. Bounds
checks compile to predication, but for power-of-two tiles you
can instead lean on `buffer_load`'s
[hardware OOB semantics](../techniques/buffer-oob-guard.md) to drop the branches.

## Fix 2 — XOR swizzle (no wasted LDS)

Padding wastes capacity and can break vectorized `ds_read_b128`/`ds_write_b128`
alignment. The alternative is a **swizzled layout** (see
[LDS swizzling](../techniques/lds-swizzling.md)): keep the tight `TILE × TILE`
storage but permute selected row bits into the column index. Using the same
mapping on store and load preserves logical correctness, but XOR being a
bijection does **not** by itself prove conflict freedom; the mask must be derived
for the actual lane mapping, opcode width, and target phase groups:

```cpp
// Schematic only: derive swizzle_bits(row) against the target phase table.
__device__ inline int sw(int row, int col) {
    return row * TILE + (col ^ swizzle_bits(row));
}
```

CK/Tensile-style GEMM prologues use such layout-specific bit permutations when
they need a transposed operand staged in LDS without paying the padding tax.

## What the inner loop looks like in GCN/CDNA assembly

The hot path is two `ds_*` instructions bracketed by a barrier. With the padded
33-/34-dword target-specific row stride, the transposed read offsets are computed once and the
`s_waitcnt lgkmcnt(0)` gates the LDS round-trip ([waitcnt](../hardware/s-waitcnt.md)):

```asm
; v2 = byte offset of tile[threadIdx.y][threadIdx.x]  (stride 33 or 34 dwords)
; v3 = byte offset of tile[threadIdx.x][threadIdx.y]  (transposed read offset)
; v4 = loaded input element (already in VGPR from global_load)
    ds_write_b32   v2, v4                ; LDS[ tile[ty][tx] ] = v4
    s_waitcnt      lgkmcnt(0)            ; LDS write visible before barrier
    s_barrier                            ; __syncthreads()
    ds_read_b32    v5, v3                ; v5 = LDS[ tile[tx][ty] ]  (transposed)
    s_waitcnt      lgkmcnt(0)            ; wait for the read to land
    ; ... global_store_dword (coalesced) of v5 ...
```

Without the target's `+1`/`+2` stride the `ds_read_b32` above would still issue
one instruction, but this mapping is 32-way conflicted per gfx942 phase and
16-way conflicted across four banks in the gfx950 wave64 phase. That
serialization is the single biggest knob on this kernel. Verify it with
`rocprofv3`/`rocprof-compute` LDS bank-conflict counters
([bank-conflicts pattern](../patterns/bank-conflicts.md)).

## Tuning notes

- **Vectorize.** Widen to `float4` per work-item and emit `ds_write_b128` /
  `ds_read_b128` to cut instruction count 4×; this favors the XOR-swizzle layout
  over padding because 128-bit LDS ops must be 16-byte aligned (a `+1`-float pad
  breaks that). See [vectorized loads](../techniques/vectorized-loads.md).
- **Rectangular thread tiles.** A `block(64,16)` with each thread handling 2 rows
  often beats `32×32`: the wave then spans 64 contiguous columns, matching the
  full coalesced cache line while keeping the LDS access pattern conflict-free.
- **gfx950 banks.** CDNA4 has 64 LDS banks. A `64×64` mapping where one wave
  reads one 64-row column can use stride `65` (`gcd(65,64)=1`); other mappings
  need their own phase-level proof. Re-derive the XOR mask as well.
- **Direct-to-LDS.** The input load can bypass VGPRs entirely with
  `buffer_load_dword ... lds` ([async copy to LDS](../hardware/async-copy-lds.md)),
  overlapping the HBM stream with the previous tile's store — useful when fusing
  transpose into a GEMM prologue.

## Performance

Transpose tops out at HBM bandwidth, so the figure of merit is **effective
bandwidth** = `2 * bytes_moved / time` measured against MI300X's 5.3 TB/s HBM3
peak ([datasheet](../../sources/docs/doc-mi300x-datasheet.md)). A well-tiled
conflict-free transpose reaches roughly **80% of peak** for large fp32 matrices;
the naive strided-store version typically lands well below half that because its
writes fragment into partial transactions. The two numbers in the frontmatter
are `inferred` from the bandwidth roofline and the bank-conflict model, not a
measured benchmark on this exact code — treat them as targets and confirm with
the [bandwidth microbenchmark](bandwidth-microbench.md) and profiler counters.

## Runnable example

A portable, self-checking version of the padded (Fix 1) kernel lives in
[`examples/transpose-lds/`](../../examples/transpose-lds/). It is **pure HIP**
(no MFMA/WMMA), builds and runs on gfx950, and also device-compiles for gfx942.
No gfx942 runtime is claimed. It transposes an fp32 matrix and requires exact
agreement with a CPU reference.

```bash
cd examples/transpose-lds && ./build.sh
```

Expected output (captured on MI355X / gfx950):

```
build: OK
Transpose 2048 x 4096 (fp32), TILE=32
avg kernel time: 0.015 ms   effective BW: 4330.7 GB/s
max abs error: 0   mismatches: 0
PASS
```

This is one mid-size run, not a tuned peak. The MI300X frontmatter figures
remain `inferred` targets.

## See also

- [LDS — banks & conflicts](../hardware/lds.md)
- [LDS swizzling technique](../techniques/lds-swizzling.md)
- [Bank-conflict avoidance](../techniques/bank-conflict-avoidance.md)
- [Bank-conflicts pattern](../patterns/bank-conflicts.md)

## Sources

- [gcnasm — AMD GCN/CDNA assembly kernel samples (transpose-lds)](https://github.com/AMD-AILlogic/gcnasm)
- [AMD CDNA3 ISA Reference Guide](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf)
- [AMD Instinct MI300X Datasheet](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/data-sheets/amd-instinct-mi300x-data-sheet.pdf)
- [LLVM AMDGPU Backend User Guide](https://llvm.org/docs/AMDGPUUsage.html)
