---
id: pattern-tail-effect
title: "Tail Effect — Wave Quantization and Idle CUs at GEMM End"
type: pattern
architectures:
- gfx942
- gfx950
tags:
- tail-effect
- idle-cu
- load-imbalance
- tile-scheduling
- stream-k
- xcd
- cu
symptoms:
- tail-effect
- idle-cu
- load-imbalance
candidate_techniques:
- technique-stream-k
- technique-split-k
- technique-persistent-kernel
related:
- pattern-low-occupancy
- pattern-xcd-locality
- technique-stream-k
- technique-split-k
- technique-persistent-kernel
- hw-chiplet-xcd
sources:
- hw-chiplet-xcd
- ref-tensile
- blog-gemm-optimization
- doc-mi300x-datasheet
- ref-composable-kernel
---

# Tail Effect — Wave Quantization and Idle CUs at GEMM End

## Symptoms

- A GEMM (or any tiled grid) achieves high SOL during its steady state but
  measured TFLOPS fall well short of the roofline, especially at **small or
  "awkward" problem sizes**.
- A profiler/trace shows most CUs finishing early while a **handful of CUs keep
  running a final "tail"** of work — the kernel's wall-clock is gated by the
  slowest CU, not the average.
- `rocprofv3`/ATT timeline shows a block of `idle-cu` time at the end of the
  dispatch; occupancy looks fine mid-kernel.
- Throughput is **non-monotonic in problem size**: e.g. M=4096 is fast, M=4097
  drops sharply, then recovers — the classic signature of **wave quantization**.

## What's actually happening

A tiled kernel launches `gridSize = ceil(M/BM) * ceil(N/BN)` workgroups (one
output tile each). The hardware runs at most `W` workgroups concurrently, where
`W` is bounded by the CU count times per-CU occupancy. The grid executes in
`ceil(gridSize / W)` **waves of workgroups**. Two things bite:

1. **Tile quantization.** If `M` or `N` is not a multiple of the macro-tile,
   the edge tiles are *partial* — they do the same MFMA work as a full tile but
   produce fewer useful results. Wasted compute on the boundary.

2. **Wave quantization (the tail).** The final wave is almost never full. If
   `gridSize % W == r` with `0 < r < W`, the last wave runs only `r` workgroups
   while `W − r` CUs sit idle, yet the kernel cannot retire until that last wave
   completes. The relative cost of the tail is `1 − (gridSize / (W·ceil(gridSize/W)))`.

The MI300X chiplet layout makes this sharper. With **304 CUs across 8 XCDs**
(38 active CUs/XCD) and L2 coherence *per-XCD* (see
[chiplet / XCD locality](../hardware/chiplet-xcd.md)), the effective wave width
is large, so a grid that isn't a clean multiple of a few hundred tiles leaves a
disproportionate tail. A 5-wave dispatch that spills 20 tiles into a 6th wave
loses ~1/6 ≈ 17% of peak even though every individual tile was efficient.

### A back-of-envelope estimate

```python
# Tail-effect efficiency for a plain data-parallel tiled GEMM.
import math

def tail_efficiency(M, N, BM, BN, cus=304, wg_per_cu=2):
    tiles  = math.ceil(M / BM) * math.ceil(N / BN)   # one workgroup per output tile
    width  = cus * wg_per_cu                          # workgroups resident at once
    waves  = math.ceil(tiles / width)                # workgroup-waves over the grid
    busy   = tiles / (waves * width)                 # fraction of slots doing work
    return tiles, waves, busy

# 4096x4096, 256x256 macro tile -> 16x16 = 256 tiles
print(tail_efficiency(4096, 4096, 256, 256))   # (256, 1, 0.42)  <-- only 256 of 608 slots used!
print(tail_efficiency(8192, 8192, 256, 256))   # (1024, 2, 0.84)
print(tail_efficiency(8192, 8192, 128, 128))   # (4096, 7, 0.96) smaller tiles -> finer grain
```

The first case is the worst kind of tail: a *single* partially-filled wave means
more than half the machine is idle for the entire kernel. This is why large
macro-tiles that look great on huge GEMMs collapse on medium ones.

## How to confirm it

- Compute `tiles`, `width`, and `waves` as above. If `tiles < width` or
  `tiles % width` is small, you are tail-bound by construction.
- In `rocprofv3` look at per-CU active time (or the ATT/Omnitrace timeline):
  a stair-step where CUs drain to idle while a few finish ⇒ tail.
- Sweep the free dimension by ±1 tile. A sharp throughput cliff at tile
  boundaries confirms wave/tile quantization rather than a memory stall.

## Candidate techniques

| Technique | When it helps | Mechanism |
|---|---|---|
| [Stream-K](../techniques/stream-k.md) | Few output tiles, large K (skinny/medium GEMM, decode) | Parallelize over the **K reduction** and assign equal-sized *work units* (not tiles) to a fixed grid of exactly `W` persistent workgroups, so every CU does the same amount of MAC work and the tail collapses. Partial sums are combined via a fixup/atomic pass. |
| [Split-K](../techniques/split-k.md) | Same shapes, simpler than Stream-K | Statically split the K dimension into `S` chunks → `S×` more workgroups to fill the last wave; reduce partials in a second kernel or with atomics. Cheap to enable but adds a reduction and extra HBM traffic. |
| [Persistent kernel](../techniques/persistent-kernel.md) | Many small tiles, launch/scheduling overhead | Launch exactly `W` workgroups that loop over a global tile counter (`atomicAdd`), amortizing launch cost and smoothing imbalance; pairs naturally with Stream-K's work decomposition and XCD-aware tile remapping. |

**First move:** pick the **macro-tile size** so the grid is a near-multiple of the
resident workgroup count (finer tiles ⇒ smaller relative tail), then apply
Stream-K when K is large enough to expose reduction parallelism. Tensile and
hipBLASLt's tuners search exactly this space — `GlobalSplitU` (split-K),
`StreamK`, and `WorkGroupMapping` are the relevant solution parameters; see
[Tensile](../../sources/refs/ref-tensile.md).

```cpp
// Persistent + atomic tile dispatch: launch gridDim.x == (resident WGs), loop tiles.
// Removes per-tile launch granularity; combine with Stream-K for the K reduction.
__global__ void persistent_gemm(const Params p, int total_tiles, int* g_tile) {
    for (int tile = atomicAdd(g_tile, 1); tile < total_tiles;
             tile = atomicAdd(g_tile, 1)) {
        int tm = tile / p.n_tiles;          // remap to (row,col); a swizzled
        int tn = tile % p.n_tiles;          // mapping improves per-XCD L2 reuse
        compute_output_tile(p, tm, tn);     // full MFMA mainloop for one tile
    }
}
```

## Trade-offs and pitfalls

- **Split-K / Stream-K add a reduction.** Partial accumulators must be combined,
  which costs HBM bandwidth and can lose a little precision if you reduce in a
  lower dtype — keep the partial sums in FP32.
- **Don't over-split.** Too many K-chunks turns a compute-bound GEMM into a
  reduction-bound one; the win from filling the tail is erased by atomic
  contention and extra passes.
- **XCD awareness matters.** Because L2 is per-XCD on MI300X, a naive linear
  tile→CU mapping scatters a tile's row/column neighbors across XCDs. Combine
  tail fixes with an XCD-friendly tile remap (see
  [XCD locality](xcd-locality.md)) to avoid trading a tail for cross-XCD L2
  misses.
- **Tail effect ≠ low occupancy.** If steady-state occupancy is already low,
  fix that first ([low occupancy](low-occupancy.md)); Stream-K assumes the
  mainloop itself is efficient.

## Sources

- [AMD CDNA3 / MI300X — chiplet & XCD topology](../hardware/chiplet-xcd.md)
- [MI300X datasheet (CU count, peak TFLOPS)](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/data-sheets/amd-instinct-mi300x-data-sheet.pdf)
- [Tensile — GEMM solution generator (StreamK, GlobalSplitU, WorkGroupMapping)](https://github.com/ROCm/Tensile)
- [Optimizing GEMM on AMD GPUs (ROCm Blogs)](https://rocm.blogs.amd.com/artificial-intelligence/gemm-optimization/README.html)
- [Composable Kernel — tile scheduling and Stream-K pipelines](https://github.com/ROCm/composable_kernel)
