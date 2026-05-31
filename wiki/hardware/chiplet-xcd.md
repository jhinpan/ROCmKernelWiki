---
id: hw-chiplet-xcd
title: XCD Chiplets, Per-XCD L2, Infinity Cache & Partition Modes (CDNA3/CDNA4)
type: hardware
architectures:
- gfx942
- gfx950
tags:
- xcd
- l2-cache
- infinity-cache
- cu
- hbm3
- cdna
- tile-scheduling
confidence: source-reported
related:
- hw-mfma
- hw-lds
- pattern-xcd-locality
- technique-persistent-kernel
- technique-stream-k
sources:
- doc-cdna3-whitepaper
- doc-cdna4-whitepaper
- doc-mi300x-datasheet
- doc-rocm-hip-hw
- blog-gemm-optimization
aliases:
- XCD
- Accelerator Complex Die
- chiplet
- NUMA
- SPX
- CPX
- partition modes
---
# XCD Chiplets, Per-XCD L2, Infinity Cache & Partition Modes (CDNA3/CDNA4)

## Overview

MI300-class accelerators are **not monolithic GPUs**. The compute silicon is
split across several **XCDs** (Accelerator Complex Dies) — chiplets stacked on
top of I/O dies that host the memory controllers and the last-level
**Infinity Cache**. Each XCD has its own pool of Compute Units (CUs) and its own
**private L2 cache**. This makes a single MI300X/MI350 behave like a small
**NUMA machine**: a CU sees its local XCD's L2 at full speed, but data produced
by a CU on another XCD must travel through the memory-side Infinity Cache /
HBM fabric to be observed coherently.

For a kernel engineer this means two things that do not exist on a monolithic
design:

1. **L2 locality is per-XCD.** Two thread blocks that share input tiles benefit
   from L2 reuse *only if the scheduler places them on the same XCD*.
2. **Block-ID → XCD mapping is round-robin**, not contiguous. Naive tiling
   scatters spatially-adjacent output tiles across all 8 XCDs, destroying reuse.

## Chiplet topology

| Part | Arch / gfx | XCDs | CUs/XCD (active/phys) | Total CUs | L2/XCD | Infinity Cache | HBM |
|---|---|---|---|---|---|---|---|
| MI300X | CDNA3 / gfx942 | 8 | 38 / 40 | 304 | 4 MB | 256 MB | 192 GB HBM3, 5.3 TB/s |
| MI350X / MI355X | CDNA4 / gfx950 | 8 | 32 / 36 | 256 | 4 MB | 256 MB | 288 GB HBM3E, up to 8 TB/s |

A few CUs per XCD are fused off for yield (40→38 on gfx942, 36→32 on gfx950),
so the *active* CU count is what occupancy math must use. The **256 MB
Infinity Cache** is a memory-side last-level cache (16-way) that sits in front of
HBM and is **shared by all XCDs** — it absorbs cross-XCD traffic and HBM
latency, but it is *not* a substitute for keeping reuse inside one XCD's 4 MB L2.

> The cache hierarchy a wavefront sees is therefore: per-CU **L1 + 64/160 kB
> [LDS](lds.md)** → per-XCD **4 MB L2** → device-wide **256 MB Infinity Cache** →
> **HBM**. Only the L2 is private to the XCD; everything above it is shared.

## Why XCD = NUMA domain

L2 coherence is maintained **per XCD**. A store from a CU is visible cheaply to
other CUs *on the same XCD* (same L2), but making it visible to a CU on a
different XCD requires the line to be written through to the coherence point
(Infinity Cache / HBM) and re-fetched. There is no global L2. Consequently:

- **Producer/consumer kernels** (e.g. split-K reduction, persistent GEMM with a
  global tile counter) pay a NUMA penalty whenever the producer and consumer
  land on different XCDs.
- **Atomics** to a shared counter that is hammered from all XCDs serialize at the
  coherence point, not in a local L2.

## Mapping block IDs to XCDs

ROCm dispatches workgroups to XCDs in a **round-robin** fashion: block 0 → XCD 0,
block 1 → XCD 1, …, block 7 → XCD 0 again (for 8 XCDs). To recover L2 locality
you can *remap* the hardware block index so that a contiguous run of tiles lands
on one XCD before moving to the next — the same idea Stream-K and persistent
GEMM schedulers use. A minimal HIP remap:

```cpp
// Remap a linear block id so that groups of tiles are XCD-local.
// num_xcd = 8 on MI300X/MI350; query at runtime, do not hardcode forever.
__device__ __forceinline__
int xcd_remap(int bid, int grid_size, int num_xcd) {
    // How many blocks each XCD will run (ceil), assuming round-robin dispatch.
    int per_xcd      = (grid_size + num_xcd - 1) / num_xcd;
    int xcd          = bid % num_xcd;   // which XCD the HW gave us
    int slot         = bid / num_xcd;   // position within that XCD
    // New logical id: all 'per_xcd' tiles of one XCD are contiguous.
    int remapped     = xcd * per_xcd + slot;
    return (remapped < grid_size) ? remapped : bid;
}

__global__ void gemm_xcd_aware(const float* A, const float* B, float* C,
                               int M, int N, int K, int num_xcd) {
    int tile = xcd_remap(blockIdx.x, gridDim.x, num_xcd);
    int tile_m = tile / ((N + 127) / 128);
    int tile_n = tile % ((N + 127) / 128);
    // ... consume A[tile_m] and B[tile_n]; tiles sharing A/B rows now
    // co-reside on one XCD's 4 MB L2 instead of scattering across all 8.
    (void)A; (void)B; (void)C; (void)M; (void)K; (void)tile_m; (void)tile_n;
}
```

The payoff is concentrating reuse of a shared operand (an A-row panel or B-column
panel) into one XCD's L2 rather than replicating it across eight L2 slices. See
[Stream-K](../techniques/stream-k.md) and
[persistent kernels](../techniques/persistent-kernel.md) for schedulers built
around this, and the [XCD-locality pattern](../patterns/xcd-locality.md) for the
symptoms (high cross-XCD traffic, low L2 hit rate) that signal you need it.

## Partition modes

MI300/MI350 can be carved up at boot/driver level along two independent axes —
**compute** and **memory** — exposed via `amd-smi` / `rocm-smi`.

**Compute partitioning (how XCDs are exposed as devices):**

| Mode | Meaning | Devices seen | Use case |
|---|---|---|---|
| **SPX** (Single Partition X-celerator) | All 8 XCDs act as one logical GPU | 1 | Large single models; max per-device CUs/HBM |
| **CPX** (Core Partition X-celerator) | Each XCD is its own logical GPU | 8 | Many small/independent jobs; tighter L2 locality per device |

In **CPX** each logical device *is* a single XCD, so the per-XCD L2 becomes "the"
L2 and the NUMA effect inside a device disappears — at the cost of 1/8th the CUs
and a memory slice per device. **SPX** gives one big device but re-introduces the
8-domain NUMA behavior described above.

**Memory partitioning (NPS — NUMA Per Socket):**

| Mode | HBM layout | Notes |
|---|---|---|
| **NPS1** | All HBM stacks as one interleaved pool | Uniform bandwidth; default for big models |
| **NPS4** | HBM split into 4 NUMA quadrants | Higher local BW per quadrant; pair with CPX for locality |

Compute and memory modes compose (e.g. `CPX` + `NPS4`). The right choice is
workload-dependent: inference servers packing many small models favor
`CPX`/`NPS4` for isolation and locality; a single large training/GEMM job favors
`SPX`/`NPS1` for one wide device. Mode is fixed for the lifetime of the
allocation — a kernel cannot change it, but it must be written to behave well
under whichever mode the operator chose.

## Practical guidance

- **Query, don't hardcode.** Get the active CU count and partition mode at
  runtime (`hipGetDeviceProperties`, `amd-smi`); CU counts differ across
  gfx942/gfx950 and shrink in CPX.
- **Cluster reuse onto one XCD.** Block-ID remapping (above) is the single
  highest-leverage XCD optimization for GEMM/attention.
- **Avoid device-wide atomic hotspots** in SPX; prefer per-XCD partial results
  reduced in a second pass, or Stream-K's bounded fix-up.
- **Tail effect is per-XCD.** With 304/256 CUs, grids that aren't a multiple of
  the active CU count leave whole XCDs idle on the last wave — see
  [tail effect](../patterns/tail-effect.md).

## See also

- [Local Data Share (LDS)](lds.md)
- [MFMA matrix cores](mfma.md)
- [XCD-locality pattern](../patterns/xcd-locality.md)
- [Stream-K scheduling](../techniques/stream-k.md)
- [Persistent kernels](../techniques/persistent-kernel.md)

## Sources

- [AMD CDNA3 Architecture Whitepaper](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-3-white-paper.pdf)
- [AMD CDNA4 Architecture Whitepaper](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-4-architecture-whitepaper.pdf)
- [AMD Instinct MI300X Datasheet](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/data-sheets/amd-instinct-mi300x-data-sheet.pdf)
- [AMD Instinct MI300/MI350 partitioning & GPU architecture (ROCm docs)](https://rocm.docs.amd.com/en/latest/conceptual/gpu-arch.html)
- [Optimizing GEMM on AMD GPUs (ROCm Blogs)](https://rocm.blogs.amd.com/artificial-intelligence/matrix-cores/README.html)
