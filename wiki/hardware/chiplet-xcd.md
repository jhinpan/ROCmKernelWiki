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
- blog-amdgpu-kernel-opt-guide
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
**NUMA machine**: a CU sees its local XCD's L2 at full speed, while traffic that
must cross an XCD boundary goes through the inter-die data fabric and
memory-side hierarchy rather than a device-wide L2.

For a kernel engineer this means two things that do not exist on a monolithic
design:

1. **L2 locality is per-XCD.** Two thread blocks that share input tiles benefit
   from L2 reuse *only if the scheduler places them on the same XCD*.
2. **Block-ID → XCD mapping is round-robin**, not contiguous. Naive tiling
   scatters spatially-adjacent output tiles across all 8 XCDs, destroying reuse.

## Chiplet topology

| Part | Arch / gfx | XCDs | I/O topology | Compute grouping/XCD | CUs/XCD (active/physical) | Total active CUs |
|---|---|---:|---|---:|---:|---:|
| MI300X | CDNA3 / gfx942 | 8 | 4 IODs, each below a pair of XCDs and attached to 2 HBM stacks | guide describes 4 Shader Engines; whitepaper directly specifies 4 ACEs and 40/38 CUs | 38 / 40 | 304 |
| MI350X / MI355X | CDNA4 / gfx950 | 8 | 2 IODs, each attached to 4 HBM stacks | four arrays of 9 physical CUs | 32 / 36 | 256 |

A few CUs per XCD are fused off for yield (40→38 on gfx942, 36→32 on gfx950),
so the *active* CU count is what occupancy math must use. On the MI355X used for
the 2026-07-20 validation pass, HIP reported 256 CUs and gfx950; this confirms
the active count, not every physical/fused-off detail in the table.

### HBM topology and peak bandwidth

| Part | Memory interfaces | Bandwidth arithmetic | Peak |
|---|---|---|---:|
| MI300X | 8 HBM3 stacks; 8192-bit aggregate interface; two stacks per IOD | `8192 bits × 5.2 Gb/s per pin ÷ 8` | 5.3248 TB/s (marketed as 5.3 TB/s) |
| MI355X | 8 HBM3E stacks; 8192-bit aggregate interface at 8 Gb/s per pin | `8192 bits × 8 Gb/s per pin ÷ 8` | 8.192 TB/s raw arithmetic; product value 8 TB/s |

Near-peak bandwidth requires balancing traffic across the HBM stacks/channels
available under the current NPS and compute-partition mode. MI300X has four IODs
and MI355X has two, so the guide's “engage all four IODs” is a MI300-specific
heuristic rather than a portable rule.

### Cache hierarchy and scope

| Level | gfx942 / MI300X | gfx950 / MI355X | Scope, policy, and evidence note |
|---|---|---|---|
| L1D | 32 KiB, 128 B line; 64-way is indirectly supported by the CDNA4 “unchanged” statement; 4 sets is derived | 32 KiB, 128 B line, 64-way; 4 sets is derived | per CU; vector-store miss/bypass behavior is more precise than treating it as an ordinary write-back cache |
| L1I | 64 KiB, 8-way | same | shared by each adjacent pair of CUs; the cited official sources do not establish the guide's 128 B line claim |
| L2 | 4 MiB/XCD = 16 × 256 KiB; 128 B line; 16-way; 128 sets/channel is derived | same capacity/line/ways; adds support for caching non-coherent DRAM data and writing back dirty lines while retaining a copy | per XCD, coherent within the XCD on both generations, write-back/write-allocate; each channel reads 128 B and can write 64 B per clock |
| MALL / Infinity Cache | 32 MiB per HBM stack × 8 = 256 MiB; 64 MiB/IOD | 32 MiB per HBM stack × 8 = 256 MiB; 128 MiB/IOD | memory-side, 16-way, 2048 sets/channel; 16 × 2 MiB channels/stack, each 64 B wide; it does not participate in lower-cache snoop/coherency traffic |

The **256 MiB Infinity Cache/MALL** is distributed as eight 32 MiB stack-local
slices. It absorbs memory-side traffic but is not a substitute for keeping reuse
within one XCD's private 4 MiB L2. The guide's “32 MiB per IOD” label is wrong:
32 MiB is **per HBM stack**, giving 64 MiB/IOD on MI300X and 128 MiB/IOD on
MI355X.

> **Correction to the guide.** AMD's gfx942 memory model says volatile vector
> and scalar **L1** lines are invalidated between dispatches; it does not say the
> entire L2 is flushed between every kernel. It also documents explicit L2
> writeback/invalidate operations. Do not rely on a launch boundary as a blanket
> L2 flush. The CDNA3/CDNA4 whitepapers do support the separate performance fact
> that per-XCD L2 coalesces traffic before it fans out to the data fabric.

> The hardware-managed data-cache path is per-CU **L1D** → per-XCD **4 MB L2**
> → device-wide **256 MB Infinity Cache** → **HBM** (with L1I shared by adjacent
> CUs). The per-CU 64/160 KiB [LDS](lds.md) is separate, explicitly managed
> workgroup storage—not a transparent cache level. Only L2 is private to an XCD;
> MALL is distributed per memory stack.

## Why an XCD is a cache-locality domain

Each XCD has its own L2, so same-XCD consumers can reuse a resident line while a
cross-XCD consumer must traverse the device fabric/coherence path; there is no
single global L2. This is a **performance-locality** statement, not an automatic
visibility guarantee. Same-XCD communication still needs the correct
synchronization, memory scope, and L1 maintenance, while correctly synchronized
cross-XCD communication need not round-trip through HBM. Consequently:

- **Producer/consumer kernels** (e.g. split-K reduction, persistent GEMM with a
  global tile counter) may pay extra fabric/coherence traffic whenever producer
  and consumer land on different XCDs.
- **Atomics** to a shared counter hammered from all XCDs cannot remain an
  XCD-local-L2 operation and can become a device-wide serialization point.

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

| Mode | Meaning | Devices seen | XCCs/device | Use case |
|---|---|---|---|---|
| **SPX** (Single Partition X-celerator) | All 8 XCDs act as one logical GPU | 1 | 8 | Large single models; max per-device CUs/HBM |
| **DPX** (Dual) | XCDs split into two halves | 2 | 4 | Two medium jobs |
| **QPX** (Quad) | XCDs split into four | 4 | 2 | Four medium jobs |
| **CPX** (Core Partition X-celerator) | Each XCD is its own logical GPU | 8 | 1 | Many small/independent jobs; tighter L2 locality per device |

All four modes are advertised by this MI350X (verified via `amd-smi partition`:
`ACCELERATOR_PARTITION_PROFILES` lists SPX/DPX/QPX/CPX = 1/2/4/8 partitions of
8/4/2/1 XCCs). In **CPX** each logical device *is* a single XCD, so the per-XCD L2 becomes "the"
L2 and the NUMA effect inside a device disappears — at the cost of 1/8th the CUs
and a memory slice per device. **SPX** gives one big device but re-introduces the
8-domain NUMA behavior described above.

**Memory partitioning (NPS — NUMA Per Socket):**

| Mode | HBM layout | Notes |
|---|---|---|
| **NPS1** | All HBM stacks as one interleaved pool | Uniform bandwidth; default for big models |
| **NPS2** | HBM split into 2 NUMA halves | Higher local BW per half; the finer split this MI350X advertises |

> Verified on this MI350X (gfx950): `amd-smi partition` reports
> `MEMORY_PARTITION_CAPS: NPS1,NPS2` (current `NPS1`); **NPS4 is not advertised
> by this gfx950 part** — it is an MI300-series (gfx942) memory layout. Older docs
> that pair `CPX` with `NPS4` describe MI300X; on MI350X the analogous fine-grained
> pairing is `CPX`/`NPS2`.

Compute and memory modes compose (e.g. `CPX` + `NPS2` on MI350X). The right choice
is workload-dependent: inference servers packing many small models favor a finer
`CPX`/`NPS2` split for isolation and locality; a single large training/GEMM job
favors `SPX`/`NPS1` for one wide device. Mode is fixed for the lifetime of the
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
- [ROCm GPU architecture specifications](https://rocm.docs.amd.com/en/latest/reference/gpu-arch-specs.html)
- [AMDGPU memory model (ROCm 7.2.1)](https://rocm.docs.amd.com/projects/llvm-project/en/docs-7.2.1/LLVM/llvm/html/AMDGPUUsage.html)
- [Optimizing GEMM on AMD GPUs (ROCm Blogs)](https://rocm.blogs.amd.com/artificial-intelligence/matrix-cores/README.html)
- [AMDGPU Kernel Optimization Guide (captured snapshot)](https://github.com/nod-ai/amd-shark-ai/blob/efa471aeef66a260c85983cc41e833bfa769dade/docs/amdgpu_kernel_optimization_guide.md) — detailed topology, bandwidth arithmetic, and cache geometry; behavioral claims are labeled above.
