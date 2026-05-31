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
- pr-composable_kernel-2177
- pr-composable_kernel-3592
- pr-composable_kernel-3038
- pr-composable_kernel-3465
- pr-composable_kernel-94
- pr-composable_kernel-3603
- pr-composable_kernel-2955
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
dword address is `(addr/4) mod num_banks`. A wave64 access is dispatched over 4
cycles at 16 lanes/cycle, so up to 16 distinct banks are serviced per cycle —
two lanes hitting the same bank in the same cycle serialize.

Store the tile naively as `float tile[TILE][TILE]` with `TILE = 32`. The
transposed read `tile[col][row]` walks **down a column**: for a fixed column,
successive rows are `TILE = 32` dwords apart, so every address maps to
`(row*32) mod 32 = 0` — **the same bank**. That is a 32-way conflict on every
transposed read, the worst case.

## Fix 1 — padding (one extra column)

Pad the row stride to `TILE + 1` dwords. Now column elements are `33` dwords
apart, and `gcd(33, 32) = 1`, so the 32 rows of a column land in 32 *distinct*
banks — conflict-free — at the cost of one wasted dword per row.

```cpp
// MI300X / MI350X — out-of-place fp32 transpose, LDS-staged, conflict-free.
// Launch: dim3 block(TILE, TILE); dim3 grid((cols+TILE-1)/TILE,(rows+TILE-1)/TILE);
#define TILE 32
__global__ void transpose_lds(float* __restrict__ out,
                              const float* __restrict__ in,
                              int rows, int cols) {
    // +1 pad column => column-wise reads hit 32 distinct banks (gcd(33,32)==1)
    __shared__ float tile[TILE][TILE + 1];

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

`block(32,32)` = 1024 work-items = **16 wave64s** per workgroup; the padded tile
costs `32*33*4 = 4224 B` of LDS, leaving plenty of the 64 kB/CU budget for high
occupancy. Bounds checks compile to predication, but for power-of-two tiles you
can instead lean on `buffer_load`'s
[hardware OOB semantics](../techniques/buffer-oob-guard.md) to drop the branches.

## Fix 2 — XOR swizzle (no wasted LDS)

Padding wastes a column and breaks vectorized `ds_read_b128`/`ds_write_b128`
alignment. The alternative is a **swizzled layout** (see
[LDS swizzling](../techniques/lds-swizzling.md)): keep the tight `TILE × TILE`
storage but permute the column index with `col ^ row`. Because XOR is a
bijection, both the write and the transposed read touch distinct banks, with no
padding overhead:

```cpp
// store: tile[row][col ^ row] = v;   load: v = tile[col][row ^ col];
__device__ inline int sw(int row, int col) { return row * TILE + (col ^ row); }
```

This is the layout CK/Tensile GEMM prologues use when they need a transposed
operand staged in LDS without paying the padding tax.

## What the inner loop looks like in GCN/CDNA assembly

The hot path is two `ds_*` instructions bracketed by a barrier. With a padded
33-dword row stride, the transposed read offsets are computed once and the
`s_waitcnt lgkmcnt(0)` gates the LDS round-trip ([waitcnt](../hardware/s-waitcnt.md)):

```asm
; v2 = byte offset of tile[threadIdx.y][threadIdx.x]  (stride 33 dwords = 132 B)
; v3 = byte offset of tile[threadIdx.x][threadIdx.y]  (transposed read offset)
; v4 = loaded input element (already in VGPR from global_load)
    ds_write_b32   v2, v4                ; LDS[ tile[ty][tx] ] = v4
    s_waitcnt      lgkmcnt(0)            ; LDS write visible before barrier
    s_barrier                            ; __syncthreads()
    ds_read_b32    v5, v3                ; v5 = LDS[ tile[tx][ty] ]  (transposed)
    s_waitcnt      lgkmcnt(0)            ; wait for the read to land
    ; ... global_store_dword (coalesced) of v5 ...
```

Without the `+1` stride the `ds_read_b32` above would still issue one
instruction, but the 32-bank serialization stretches it to ~32× its
conflict-free latency — the single biggest knob on this kernel. Verify it with
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
- **gfx950 banks.** CDNA4 has 64 LDS banks. A `64×64` tile needs stride `65`
  (`gcd(65,64)=1`) for the padded variant; the XOR-swizzle variant is unchanged.
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
(no MFMA/WMMA), so it builds and runs natively on gfx1201 (RDNA4) as well as
gfx942/gfx950. It transposes an fp32 matrix, verifies the result is *exactly*
equal to a CPU reference, and reports effective bandwidth.

```bash
cd examples/transpose-lds
hipcc --offload-arch=gfx1201 -O3 transpose_lds.cpp -o transpose_lds && ./transpose_lds
```

Expected output (captured on an RX 9070 XT / gfx1201, ROCm 7.2.3):

```
Transpose 2048 x 4096 (fp32), TILE=32
avg kernel time: 0.138 ms   effective BW: 487.2 GB/s
max abs error: 0   mismatches: 0
PASS
```

The reported bandwidth is real gfx1201 timing for one mid-size matrix (not a
tuned peak); the MI300X figures in the frontmatter remain `inferred` targets.

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
