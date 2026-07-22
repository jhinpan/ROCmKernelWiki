---
id: technique-persistent-kernel
title: Persistent Kernels — Grid-Stride Work Loops for Launch & L2 Reuse
type: technique
architectures:
- gfx942
- gfx950
tags:
- persistent-kernel
- tile-scheduling
- flat-work-scheduling
- occupancy-tuning
- xcd
- l2-cache
- stream-k
confidence: source-reported
reproducibility: snippet
hardware_features:
- xcd
- l2-cache
- cu
kernel_types:
- gemm
- bandwidth-bench
- elementwise
languages:
- hip
- gcn-asm
related:
- technique-stream-k
- technique-split-k
- technique-occupancy-tuning
- hw-chiplet-xcd
- kernel-bandwidth-microbench
- pattern-tail-effect
sources:
- ref-gcnasm
- hw-chiplet-xcd
- kernel-bandwidth-microbench
- blog-gemm-optimization
- doc-mi300x-datasheet
implemented_by:
- pr-composable_kernel-3202
- pr-composable_kernel-1360
- pr-composable_kernel-3520
- pr-composable_kernel-3359
- pr-composable_kernel-3107
- pr-composable_kernel-2786
- pr-composable_kernel-2781
- pr-composable_kernel-2721
---
# Persistent Kernels — Grid-Stride Work Loops for Launch & L2 Reuse

## Overview

A **persistent kernel** launches exactly enough workgroups (WGs) to fill the
machine — typically one WG per CU (or a small integer multiple) — and then has
each WG loop over many logical work tiles instead of mapping one tile to one WG.
The grid is *resident* for the whole problem: WGs are launched once, stay live,
and pull tiles from a shared counter or a strided index space until the work is
exhausted.

Contrast the two scheduling models:

- **One-tile-per-WG (default):** grid size = number of tiles. The hardware
  dispatcher streams WGs onto CUs as slots free up. Simple, but every tile pays
  WG-launch/teardown overhead, and the *tail* (the last incomplete wave of WGs)
  leaves CUs idle — see [tail effect](../patterns/tail-effect.md).
- **Persistent:** grid size = number of CUs × WGs-per-CU. Each WG runs a
  `for (tile = wg_id; tile < num_tiles; tile += grid_size)` loop. Launch cost is
  paid once, registers/LDS are allocated once, and the WG↔CU mapping is stable
  for the kernel's lifetime — which the programmer can exploit for **L2 locality**.

On MI300X that resident grid is 304 CUs across 8 XCDs; on MI355X it is 256 CUs
across 8 XCDs (see the [chiplet / XCD page](../hardware/chiplet-xcd.md)). Because
the L2 is **per-XCD**, *which* CU a tile lands on determines *which* L2 it reuses
— making the persistent mapping a tuning knob, not just an overhead reducer.

## Why it helps

1. **Amortized launch & setup.** WG dispatch, SGPR/VGPR/AGPR allocation, V#
   descriptor setup, and LDS base init happen once per WG rather than once per
   tile. For many small tiles this is a measurable fraction of runtime.
2. **Tail-effect mitigation.** With a fixed grid, the last tiles are absorbed by
   the same resident WGs instead of forming a short, under-occupied final wave.
   This is the foundation [Stream-K](stream-k.md) builds on to balance partial
   tiles across CUs.
3. **L2 / Infinity-Cache reuse.** A persistent WG that processes a *spatially
   coherent* set of tiles (e.g. a block of GEMM output rows) re-reads the same A
   or B panels from its XCD-local 4 MB L2 instead of re-fetching from HBM. Pinning
   tile groups to XCDs turns the per-XCD L2 into a working-set cache.
4. **Stable state across tiles.** Accumulators, prefetch buffers, and even
   software pipelines can carry across iterations of the persistent loop.

## Minimal HIP persistent loop

The canonical pattern: launch `gridDim = num_CUs * wgs_per_cu`, then grid-stride
over tiles. A device-wide atomic counter gives dynamic load balancing; a plain
strided index gives static, branch-free balancing.

```cpp
// Launch with: dim3 grid(num_cu * wgs_per_cu), block(256);
// Query num_cu at runtime — do NOT hardcode (304 on MI300X, 256 on MI355X).
//   hipDeviceProp_t p; hipGetDeviceProperties(&p, dev);
//   int num_cu = p.multiProcessorCount;
extern "C" __global__ void persistent_axpy(
    const float* __restrict__ x, const float* __restrict__ y,
    float* __restrict__ out, float a, int num_tiles, int tile_elems)
{
    const int grid_size = gridDim.x;          // resident WG count
    const int lane      = threadIdx.x;
    const int nthreads  = blockDim.x;

    // Static grid-stride over logical tiles: no atomics, fully branchless.
    for (int tile = blockIdx.x; tile < num_tiles; tile += grid_size) {
        const int base = tile * tile_elems;
        for (int i = lane; i < tile_elems; i += nthreads) {
            out[base + i] = a * x[base + i] + y[base + i];
        }
    }
}
```

For irregular per-tile cost, replace the static stride with a pulled counter so
fast WGs steal more work:

```cpp
__device__ int g_next_tile;   // zero-initialized before launch

extern "C" __global__ void persistent_dynamic(/* ... */ int num_tiles)
{
    __shared__ int s_tile;
    for (;;) {
        if (threadIdx.x == 0) s_tile = atomicAdd(&g_next_tile, 1);
        __syncthreads();                 // -> s_barrier
        int tile = s_tile;
        if (tile >= num_tiles) break;
        // ... process tile ...
        __syncthreads();
    }
}
```

Keep `wgs_per_cu` consistent with the kernel's occupancy: if VGPR/LDS limits
allow only 2 waves/CU, launching 8 WGs/CU just oversubscribes and re-introduces
dispatch churn. Tune it alongside [occupancy](occupancy-tuning.md).

## XCD-aware tile remapping (CDNA3/CDNA4)

Because the dispatcher assigns `blockIdx.x` to CUs in a way that interleaves
across XCDs, naive contiguous tiles spread a panel's reuse across *all 8* L2
slices. A simple remap clusters consecutive logical tiles onto the same XCD so
that an A/B panel stays hot in one 4 MB L2:

```cpp
// 8 XCDs on MI300X / MI350. Remap so that tiles that share operands
// land on CUs of the same XCD (per-XCD L2 = effective NUMA domain).
__device__ inline int xcd_remap(int tile, int num_tiles, int num_xcd /*=8*/) {
    int per_xcd = (num_tiles + num_xcd - 1) / num_xcd;
    int xcd     = blockIdx.x % num_xcd;        // which XCD this WG sits on
    int slot    = tile / num_xcd;              // round-robin position
    int mapped  = xcd * per_xcd + slot;
    return (mapped < num_tiles) ? mapped : tile;
}
```

This is the same idea Stream-K and tuned hipBLASLt/CK GEMM schedules use; the
[GEMM optimization guide](../../sources/blogs/blog-gemm-optimization.md) shows the
L2-hit-rate gains from XCD-coherent tiling. Measure with `rocprofv3`
(`L2CacheHit`, `FETCH_SIZE`) rather than assuming — gains depend on panel size
versus the 4 MB/XCD L2.

## The gcnasm persistent pattern

AMD's [`gcnasm` examples](../../sources/refs/ref-gcnasm.md) include a persistent
assembly kernel that pairs the resident-grid loop with **direct-to-LDS async
copies** and `s_waitcnt`-gated software pipelining: one WG per CU streams the
whole input, issuing `buffer_load ... lds` for the next tile while the matrix/ALU
work on the current tile completes. The persistent loop is what makes the prefetch
worthwhile — the steady-state pipeline is established once and amortized over many
tiles, which is exactly how the
[float4 bandwidth microbenchmark](../kernels/bandwidth-microbench.md) approaches
HBM peak (≈5.3 TB/s on MI300X).

```asm
; sketch: persistent steady-state loop body (one WG/CU)
loop_tile:
    buffer_load_dwordx4  v[0:3], v_off, s[desc:desc+3], 0 offen   ; stream current tile
    s_waitcnt            vmcnt(0)
    ; ... consume v[0:3] (reduce / mac / store) ...
    v_add_u32            v_tile, v_tile, s_grid_stride            ; tile += grid_size
    v_cmp_lt_u32         vcc, v_tile, s_num_tiles
    s_cbranch_vccnz      loop_tile
```

## When NOT to use it

- **Already compute-bound with a full grid.** If the default grid covers the
  machine many times over and tiles are uniform, the dispatcher hides launch cost
  for free; a persistent rewrite adds index arithmetic for no gain.
- **Dynamic atomic counter on a memory-bound kernel.** The `atomicAdd` traffic
  can serialize and cost more than the tail it removes; prefer static grid-stride
  unless per-tile cost truly varies (e.g. MoE / [grouped GEMM](../kernels/grouped-gemm.md)).
- **Correctness traps:** a device-global counter must be reset between launches,
  and any cross-tile reuse of LDS/accumulators needs a `__syncthreads()` barrier
  between iterations. OOB tiles still need [buffer OOB guards](buffer-oob-guard.md).

## Performance notes

- Launch amortization is largest for **many small tiles**; for a handful of large
  tiles the win is dominated by tail-effect and L2 reuse instead.
- The persistent grid is a *prerequisite* for [Stream-K](stream-k.md) and for
  [split-K](split-k.md) reduction schemes that need a stable WG↔CU mapping.
- Treat `num_cu`, `wgs_per_cu`, and the XCD remap as the three tunables; sweep
  them per shape and confirm with profiler L2-hit and occupancy counters.

## See also

- [Stream-K tile scheduling](stream-k.md)
- [Split-K reduction](split-k.md)
- [Occupancy tuning](occupancy-tuning.md)
- [XCD chiplet architecture & per-XCD L2](../hardware/chiplet-xcd.md)
- [Tail-effect pattern](../patterns/tail-effect.md)

## Sources

- [gcnasm — AMD GCN/CDNA assembly examples (persistent kernel)](https://github.com/ROCm/amd_matrix_instruction_calculator) — see `ref-gcnasm`
- [AMD GEMM optimization on CDNA](https://rocm.blogs.amd.com/artificial-intelligence/matrix-cores/README.html)
- [AMD Instinct MI300X datasheet (304 CUs, 8 XCDs, 5.3 TB/s)](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/data-sheets/amd-instinct-mi300x-data-sheet.pdf)
