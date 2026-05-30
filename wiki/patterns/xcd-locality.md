---
id: pattern-xcd-locality
title: "Poor XCD / L2 Locality and Cross-Chiplet Traffic (CDNA3/CDNA4)"
type: pattern
architectures:
- gfx942
- gfx950
tags:
- xcd
- l2-cache
- infinity-cache
- numa
- cdna
- persistent-kernel
- tile-scheduling
- stream-k
symptoms:
- poor-l2-locality
- cross-xcd-traffic
- numa-imbalance
candidate_techniques:
- technique-stream-k
- technique-persistent-kernel
related:
- hw-chiplet-xcd
- technique-stream-k
- technique-persistent-kernel
- pattern-tail-effect
- pattern-memory-bound
- kernel-ck-hgemm
sources:
- hw-chiplet-xcd
- doc-cdna3-whitepaper
- doc-cdna4-whitepaper
- doc-mi300x-datasheet
- blog-gemm-optimization
---

# Poor XCD / L2 Locality and Cross-Chiplet Traffic

## Symptom

A kernel that *should* be re-using data out of L2 is instead hammering HBM, and
profiling shows:

- **`poor-l2-locality`** — low L2 hit rate despite a working set that fits in
  aggregate L2; the same tensor tile is fetched from HBM many times.
- **`cross-xcd-traffic`** — high Infinity Cache (LLC) / fabric read traffic and
  L2 miss latency that looks like remote access, not local-XCD access.
- **`numa-imbalance`** — some XCDs (or HBM stacks) are saturated while others are
  idle; throughput scales poorly past one chiplet's worth of CUs.

On MI300X/MI325X (CDNA3) and MI350X/MI355X (CDNA4) this is a **chiplet
locality** problem, not a classic cache-blocking problem. The tiles may be sized
correctly; they are simply landing in the wrong cache.

## Why it happens (the hardware model)

MI300-class GPUs are multi-die. An MI300X is **8 XCDs (Accelerator Complex
Dies)**, each with 38 active CUs (40 physical) for 304 CUs total; an MI355X is
**8 XCDs × 32 active CUs = 256 CUs**. Critically:

- **L2 is private per XCD** (4 MB per XCD on both CDNA3 and CDNA4). There is no
  shared, GPU-wide L2.
- The only cache shared across XCDs is the **256 MB Infinity Cache** (the
  memory-side last-level cache) and then HBM itself (192 GB / 5.3 TB/s on
  MI300X; 288 GB / up to 8 TB/s on MI355X).

So **L2 coherence and reuse are per-XCD: the XCD is effectively a NUMA domain.**
If two workgroups that touch the same tile are scheduled onto *different* XCDs,
each pays a full HBM/LLC fetch and caches a private copy in its own 4 MB L2 —
the reuse you expected never materializes.

The second trap is the **default block-to-XCD mapping**. The hardware dispatches
workgroups round-robin across XCDs by block index: with 8 XCDs, blocks
`0,8,16,…` go to XCD0, `1,9,17,…` to XCD1, and so on (in SPX mode). A naïve
GEMM that walks output tiles in row-major `blockIdx` order therefore *scatters*
tiles that share an A-row or B-column across all 8 XCDs, maximizing cross-chiplet
refetching.

```
A row reused by tiles t0..t7 (same block-row)
default round-robin:  t0→XCD0  t1→XCD1 ... t7→XCD7   ← A row fetched 8×
XCD-aware grouping:    t0..t7 → XCD0                   ← A row fetched 1× into L2
```

## How to confirm it

Use `rocprofv3` / `rocprof-compute` (Omniperf) and look at L2 and fabric
counters rather than raw occupancy:

```bash
# Per-XCD L2 hit/miss + fabric traffic for a GEMM
rocprofv3 --pmc TCC_HIT TCC_MISS TCC_EA_RDREQ TCC_EA_WRREQ \
    -- ./my_gemm

# Quick sanity: pin the process to one XCD with CPX partitioning and compare.
# If single-XCD L2 hit rate >> full-chip hit rate, you have a locality problem.
```

Indicators that point specifically at XCD locality (not generic cache pressure):

- L2 hit rate climbs sharply when you restrict the launch to a single XCD
  (e.g. CPX compute-partition mode), even though per-XCD L2 is unchanged at 4 MB.
- `TCC_EA_RDREQ` (requests to the EA/fabric, i.e. L2 misses going off-XCD) is
  high relative to compute, and the working set is < total L2.
- Throughput plateaus around 1/8 of peak when the kernel is memory-bound on a
  hot tensor.

## Candidate techniques

### 1. XCD-aware tile scheduling via a persistent kernel

Launch exactly one persistent workgroup per CU (a
[persistent kernel](../techniques/persistent-kernel.md)) and **remap the linear
tile id so that consecutive tiles that share operands land on the same XCD.**
Because the hardware assigns `blockIdx.x % num_XCD` to an XCD, you can invert
that mapping in software: group the work so each XCD owns a contiguous *band* of
output tiles whose A-rows/B-columns stay resident in that XCD's 4 MB L2.

```cpp
// XCD-aware remap for an MxN tiled GEMM on MI300X (8 XCDs).
// Goal: tiles that reuse the same A/B data resolve to the SAME XCD,
// so their shared operands stay hot in that XCD's private 4 MB L2.
constexpr int NUM_XCD = 8;            // MI300X/MI355X: 8 XCDs

__device__ int remap_tile(int linear_tile, int num_tiles) {
    // Hardware places block b on XCD (b % NUM_XCD). We want a *contiguous*
    // chunk of tiles per XCD, so undo the round-robin interleave:
    int tiles_per_xcd = (num_tiles + NUM_XCD - 1) / NUM_XCD;
    int xcd  = linear_tile % NUM_XCD;        // which XCD this block runs on
    int slot = linear_tile / NUM_XCD;        // position within that XCD
    int tile = xcd * tiles_per_xcd + slot;   // contiguous band per XCD
    return (tile < num_tiles) ? tile : -1;   // -1 => no work (tail)
}

extern "C" __global__ void persistent_gemm(/* A,B,C, dims */ int num_tiles) {
    // One workgroup per CU; loop over this XCD's band of tiles.
    for (int t = blockIdx.x; t < num_tiles; t += gridDim.x) {
        int tile = remap_tile(t, num_tiles);
        if (tile < 0) continue;
        // ... compute output tile `tile`; its A-row / B-col tiles are reused
        //     by neighbouring tiles on the SAME XCD and stay resident in L2.
    }
}
```

Order the band so the reused operand (e.g. the A panel for a block-row) is shared
by adjacent tiles — a grouped / "super-tile" launch order. This is the same
rasterization idea as CUDA threadblock swizzling, but the unit you are localizing
to is the **XCD's L2**, not a single SM's L1.

### 2. Stream-K with XCD-aware partitioning

When tile counts don't divide evenly across 304/256 CUs you also hit the
[tail effect](tail-effect.md). [Stream-K](../techniques/stream-k.md) splits the
K dimension across CUs to keep every CU busy — but a naïve Stream-K worsens
locality by spraying partial tiles everywhere. Combine the two: assign
**contiguous output-tile bands to each XCD**, then apply Stream-K *within* an XCD
so the K-splits of a given output tile reduce inside one XCD's L2 before the
final atomic/partials fixup. You get tail-free CU utilization *and* keep the
shared A/B panels local.

### 3. Partition modes as a coarse fallback

If you cannot change the kernel, the **compute/memory partition modes** exposed
by the firmware are a blunt instrument that helps: **CPX** carves the GPU into
per-XCD compute partitions (each a NUMA-clean domain), and **NPS4** narrows each
partition to its local HBM stacks. For a single large kernel that genuinely needs
all CUs, stay in **SPX/NPS1** and fix scheduling in software (techniques 1–2);
for many small concurrent kernels, CPX can recover locality without code changes.
See [chiplet & XCD architecture](../hardware/chiplet-xcd.md) for the mode matrix.

## Pitfalls

- **Don't hardcode `NUM_XCD = 8` blindly.** It is 8 on MI300X and MI355X, but
  query the partition mode — under CPX a "device" is a single XCD, so the round
  trip changes. Detect at runtime rather than assuming SPX.
- **L2 is only 4 MB per XCD.** Localizing tiles to an XCD only helps if the
  *per-XCD* reuse set fits in 4 MB. If your tile band is too wide it spills to
  Infinity Cache anyway — tune band width against 4 MB, not against the 256 MB
  LLC.
- **Persistent kernels raise register/LDS pressure** (the outer loop and
  bookkeeping live for the kernel's lifetime). Watch occupancy; a locality win
  can be eaten by a [VGPR-pressure](vgpr-pressure.md) regression.
- **Atomics in Stream-K cross XCDs are expensive** because they serialize through
  the coherence point. Keep partial-sum reductions within an XCD where possible.

## Sources

- [AMD CDNA3 Architecture Whitepaper](https://www.amd.com/content/dam/amd/en/documents/instinct-business-docs/white-papers/amd-cdna-3-white-paper.pdf)
- [AMD CDNA4 Architecture Whitepaper](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-4-architecture-whitepaper.pdf)
- [AMD Instinct MI300X Datasheet](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/data-sheets/amd-instinct-mi300x-data-sheet.pdf)
- [Optimizing GEMM kernels on AMD GPUs (ROCm Blogs)](https://rocm.blogs.amd.com/artificial-intelligence/matrix-cores/README.html)
