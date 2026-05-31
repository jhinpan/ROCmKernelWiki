---
id: technique-stream-k
title: Stream-K — Load-Balanced GEMM Tile Scheduling for CUs/XCDs
type: technique
architectures:
- gfx942
- gfx950
tags:
- stream-k
- tile-scheduling
- split-k
- persistent-kernel
- tail-effect
- xcd
- gemm
confidence: source-reported
reproducibility: snippet
hardware_features:
- xcd
- cu
- l2-cache
kernel_types:
- gemm
- hgemm
- fp8-gemm
languages:
- hip
- composable-kernel
- triton
related:
- technique-split-k
- technique-persistent-kernel
- hw-chiplet-xcd
- kernel-ck-hgemm
- pattern-tail-effect
- pattern-xcd-locality
sources:
- hw-chiplet-xcd
- ref-composable-kernel
- blog-gemm-optimization
- doc-mi300x-datasheet
- ref-tensile
implemented_by:
- pr-Tensile-1815
- pr-composable_kernel-933
- pr-composable_kernel-2059
- pr-composable_kernel-1862
- pr-FlyDSL-340
- pr-composable_kernel-882
- pr-composable_kernel-3237
- pr-composable_kernel-2152
---
# Stream-K — Load-Balanced GEMM Tile Scheduling for CUs/XCDs

## The problem: quantization tail on a chiplet GPU

A classic "data-parallel" GEMM assigns one output tile (`BM×BN`) to one
workgroup and loops over the full `K` dimension inside that workgroup. The grid
therefore has exactly `ceil(M/BM) * ceil(N/BN)` workgroups. The trouble is that
this count rarely divides evenly into the number of physical compute units.

On MI300X there are **304 CUs across 8 XCDs** (38 active CUs/XCD); MI355X has
**256 CUs across 8 XCDs** (32/XCD) — see [chiplet/XCD](../hardware/chiplet-xcd.md).
If a problem produces, say, 320 tiles, the hardware runs one full wave of ~304
tiles, then a second wave of only 16 tiles while the other ~288 CUs sit idle.
That trailing partial wave is the **quantization / tail effect**
([pattern](../patterns/tail-effect.md)): for many practically sized GEMMs the tail
can cost 20–50% of wall-clock time because the last wave is almost as expensive
as a full one but utilizes a fraction of the machine.

**Split-K** ([technique](split-k.md)) helps the opposite case — too few tiles,
each with a long `K` — by cutting `K` into chunks computed by separate
workgroups and summing the partials. But split-K uses a *fixed* split factor, so
it just re-quantizes the work at a different granularity and needs its own
reduction pass. **Stream-K** generalizes the idea: it assigns a *fraction of the
total MAC work* to each workgroup so that every CU does (almost) exactly the same
amount of work, eliminating the tail by construction.

## The Stream-K idea

Flatten the GEMM into a single 1-D iteration space over **MAC-loop iterations**.
Each output tile needs `iters_per_tile = ceil(K / BK)` K-loop iterations, and the
total work is:

```
total_iters = num_output_tiles * iters_per_tile
```

Launch a **persistent grid** ([technique](persistent-kernel.md)) of exactly
`num_CUs` workgroups (one resident per CU). Give each workgroup a contiguous,
equal-sized slice of `total_iters`:

```
iters_per_cu = ceil(total_iters / num_CUs)
my_start = cu_id * iters_per_cu
my_end   = min(my_start + iters_per_cu, total_iters)
```

A slice generally **starts in the middle of one output tile and ends in the
middle of another**. So a workgroup accumulates a *partial* result for the tile
it starts on, and one or more whole/partial tiles after it. Whenever a slice
boundary falls inside a tile, two or more workgroups have each computed a partial
sum for that tile and must combine them — a **fix-up / reduction** step. The
workgroup that owns the *first* iteration of a tile is responsible for the final
accumulation (commonly via an atomic add into the C tile, or a small per-tile
spin-lock + LDS reduction to keep numerics deterministic).

```
tile 0        tile 1        tile 2        tile 3
|====|====|  |====|====|  |====|====|  |====|====|   (iters_per_tile = 4)
^----- WG0 -----^----- WG1 -----^----- WG2 -----^     (equal iter slices)
       partial of tile1 ^   ^ partial of tile1
              -> WG0 and WG1 both touch tile1 -> fix-up
```

## Minimal scheduler (HIP)

The scheduling math is what matters; the inner MFMA tile loop is identical to a
normal GEMM ([CK HGEMM](../kernels/ck-hgemm.md)). This snippet shows a
"Stream-K + data-parallel" hybrid, the form most real libraries ship.

```cpp
// Launch: gridDim.x = numCUs (persistent), blockDim.x = 256 (wave64 x4)
__global__ void streamk_gemm(const half* __restrict__ A,
                             const half* __restrict__ B,
                             float* __restrict__ C,
                             float* __restrict__ partials,   // workspace
                             int* __restrict__ tile_locks,   // 1 per tile
                             int M, int N, int K,
                             int itersPerTile, int totalIters,
                             int itersPerCU)
{
    const int cu = blockIdx.x;
    int it = cu * itersPerCU;                 // my first global iteration
    const int itEnd = min(it + itersPerCU, totalIters);

    while (it < itEnd) {
        const int tile      = it / itersPerTile;          // which output tile
        const int kIterInTile = it % itersPerTile;        // where I enter it
        // contiguous run of iterations I own *within this tile*
        const int tileLast  = min((tile + 1) * itersPerTile, itEnd);
        const int kIters    = tileLast - it;              // iters I contribute

        // --- map `tile` to (m,n); remap so consecutive tiles stay on one XCD ---
        int m, n;
        tile_to_mn(tile, M, N, &m, &n);

        // --- accumulate kIters of the MAC loop with v_mfma_* (omitted) ---
        float acc[ACC_REGS];
        mac_loop(A, B, m, n, /*kStart=*/kIterInTile, kIters, acc);

        const bool ownsTile = (kIterInTile == 0);
        const bool wholeTile = ownsTile && (kIters == itersPerTile);

        if (wholeTile) {
            store_tile(C, m, n, acc);                 // pure data-parallel tile
        } else {
            // partial: reduce across the WGs that share this tile
            streamk_fixup(C, partials, tile_locks, tile, m, n, acc, ownsTile);
        }
        it = tileLast;                                // advance to next tile
    }
}
```

The two design choices that dominate performance are (1) how partials are
combined and (2) how `tile_to_mn` maps the flat tile index to `(m, n)`.

## XCD-aware tile mapping

L2 is **per-XCD** on MI300/MI350, so the XCD is effectively a NUMA domain
([chiplet/XCD](../hardware/chiplet-xcd.md), [locality pattern](../patterns/xcd-locality.md)).
Two tiles that share rows of `A` or columns of `B` should land on the **same
XCD** so the shared operand is served from one L2 instead of being replicated
across eight. Because the hardware launches workgroups round-robin across XCDs
(workgroup `i` → XCD `i % numXCD`), a naive row-major tile index scatters a tile
row across all eight XCDs.

A common fix is to **rasterize tiles in XCD-sized super-groups** so that the
`numXCD` consecutive workgroups that the dispatcher fans out to all map to
neighbouring tiles:

```cpp
// Remap so that the round-robin XCD assignment keeps an L2-friendly tile group.
__device__ void tile_to_mn(int tile, int M, int N, int* m, int* n) {
    const int NUM_XCD = 8;                 // MI300X / MI355X
    const int tilesN  = (N + BN - 1) / BN;
    // pull tiles into groups of NUM_XCD, then linearize group-major
    int grp   = tile / NUM_XCD;
    int lane  = tile % NUM_XCD;
    int remapped = lane * ((/*numTiles*/ 0) ) /*…*/ ; // library-specific swizzle
    int t = grp * NUM_XCD + lane;          // simplified; real code swizzles grp
    *m = (t / tilesN) * BM;
    *n = (t % tilesN) * BN;
}
```

Combined with **column/grouped rasterization** (also used for plain
data-parallel GEMM in the [GEMM optimization blog](https://rocm.blogs.amd.com/artificial-intelligence/gemm-optimization/README.html)),
this keeps the working set of an XCD inside its 4 MB L2 and feeds the rest from
the 256 MB Infinity Cache.

## When to use it

- **Use Stream-K** for irregular / small-to-medium GEMM shapes where the tile
  count is not a clean multiple of the CU count, especially "tall-skinny" or
  "fat-K" problems (decode GEMV-like, LoRA, attention projections). It removes
  the tail without you having to hand-pick a split factor.
- **Prefer plain data-parallel** when the tile count is already a large multiple
  of `num_CUs` (big training GEMMs): the fix-up atomics are pure overhead there.
- **Prefer fixed Split-K** ([technique](split-k.md)) only when you need a simple,
  deterministic reduction and the shape is uniformly fat-K.
- The hybrid in the snippet — Stream-K for the first few "ragged" waves, then
  data-parallel for the bulk — is what production libraries
  ([Composable Kernel](https://github.com/ROCm/composable_kernel),
  [Tensile](../../sources/refs/ref-tensile.md), and the Triton AMD backend via
  `tl.dot` + persistent loop) actually ship.

## Pitfalls

- **Atomic contention / numerics.** Atomic-add fix-up is fast but
  nondeterministic in FP. Use a workspace + per-tile lock and an ordered LDS
  reduction when bit-reproducibility matters.
- **Workspace sizing.** Partial-tile reduction needs scratch of up to
  `num_CUs * BM * BN` accumulator elements; size it for the worst case or fall
  back to atomics.
- **Persistent occupancy.** A persistent grid pins one workgroup per CU, so VGPR
  ([budgeting](vgpr-budgeting.md)) and LDS use must allow at least one resident
  wave-group per CU or you lose the very parallelism you launched for.
- **`s_waitcnt` on the fix-up path.** The lock/partial load uses VMEM atomics;
  gate the dependent reduction with `vmcnt` before consuming the partial.

## Sources

- [AMD chiplet / XCD architecture (this wiki)](../hardware/chiplet-xcd.md) — per-XCD L2, CU counts, NUMA mapping.
- [Composable Kernel (ROCm/composable_kernel)](https://github.com/ROCm/composable_kernel) — Stream-K and grouped GEMM tile schedulers.
- [GEMM optimization on AMD GPUs (ROCm Blogs)](https://rocm.blogs.amd.com/artificial-intelligence/gemm-optimization/README.html) — tile rasterization and L2 locality.
- [AMD Instinct MI300X datasheet](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/data-sheets/amd-instinct-mi300x-data-sheet.pdf) — 304 CU / 8 XCD topology, peak FLOPS.
- [Tensile GEMM generator (ROCm/Tensile)](https://github.com/ROCm/Tensile) — production Stream-K / data-parallel solution selection.
- Osama et al., "Stream-K: Work-centric Parallel Decomposition for Dense Matrix-Matrix Multiplication," PPoPP 2023 (original algorithm).
