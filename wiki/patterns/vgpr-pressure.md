---
id: pattern-vgpr-pressure
title: VGPR Pressure, Register Spills, and Occupancy Collapse
type: pattern
version_sensitive:
- vs-wave-slots-cdna3-cdna4
- vs-cdna-unified-vgpr-agpr-allocation
architectures:
- gfx942
- gfx950
tags:
- vgpr-pressure
- register-spill
- low-occupancy
- vgpr
- agpr
- wave64
- occupancy-tuning
- vgpr-budgeting
symptoms:
- vgpr-pressure
- register-spill
- low-occupancy
candidate_techniques:
- technique-vgpr-budgeting
- technique-occupancy-tuning
related:
- hw-wavefront
- hw-mfma
- technique-vgpr-budgeting
- technique-occupancy-tuning
- pattern-low-occupancy
sources:
- hw-wavefront
- hw-mfma
- doc-rocm-hip-hw
- doc-llvm-amdgpu
- blog-gemm-optimization
implemented_by:
- pr-Tensile-1371
- pr-Tensile-1100
- pr-Tensile-1383
---
# VGPR Pressure, Register Spills, and Occupancy Collapse

## The pattern

A kernel runs much slower than its arithmetic intensity predicts, the profiler
shows few waves resident per CU, and latency from a memory miss is poorly hidden
because there are few other ready waves to switch to. A frequent root cause is
that **each wave consumes too many vector registers**, so the CU cannot
keep enough waves in flight to cover memory and MFMA latency. In the worst case
the compiler runs out of registers entirely and inserts **spill/reload traffic
to scratch** (private memory backed by HBM), which is both slow and consumes
even more VMEM bandwidth.

On CDNA the VGPR file is the scarcest occupancy resource for most compute-bound
kernels. CDNA3/CDNA4 provide one combined **512-entry-per-lane vector capacity
per SIMD** for regular ArchVGPR and accumulator AccVGPR/AGPR allocations. Each
view has up to 256 names; the combined allocation is encoded in **groups of 8
dwords**. HSA metadata `.vgpr_count` is already the combined total on gfx942
and gfx950, so it is rounded to 8 directly. CDNA3/CDNA4 support **up to 32 waves
(4 SIMD pools × 8 waves)** per CU, with the vector-register limit computed as
`floor(512 / vector_alloc)`, capped at 8 waves/SIMD. See
[wavefront & occupancy](../hardware/wavefront.md).

## Symptoms and how to confirm them

| Symptom | Where you see it | What it means |
|---|---|---|
| `vgpr-pressure` | High combined vector allocation in the compiler resource report | Per-wave allocation is large, capping occupancy |
| `register-spill` | Non-zero spill counts/private segment plus `scratch_load/store` in ISA | Live values or explicit private data use HBM-backed scratch |
| `low-occupancy` | `rocprofv2`/`rocprof` `Wavefronts` or `Occupancy` counters far below peak | Too few waves to hide latency |

Fast triage from the build itself — no profiler needed:

```bash
# Per-kernel resource usage: VGPR/AGPR/SGPR/LDS and SCRATCH (spills)
hipcc -O3 --offload-arch=gfx942 -Rpass-analysis=kernel-resource-usage \
      -c gemm.hip -o gemm.o

# Or dump the AMDGPU kernel metadata and grep the budget
llvm-objdump --arch-name=amdgcn -d gemm.o | head -50
# Look for:  .vgpr_count / .agpr_count / .sgpr_count / .private_segment_fixed_size
```

On both targets, round metadata `.vgpr_count` to eight and do not double-count
its `.agpr_count` subset. Separate low-level compiler `NumVgprs`/`NumAgprs`
remarks use the target-specific derivation documented on the
[wavefront page](../hardware/wavefront.md). A non-zero
`.private_segment_fixed_size` means scratch/private storage exists, but use
spill counts and emitted `scratch_*` instructions to distinguish register
spills from explicit private objects. Hot-path scratch is a red flag; the first
goal is usually to get actual spill traffic to **zero**, then tune occupancy.

## Why it happens in CDNA MFMA kernels

The following AGPR and wave64 details are specific to gfx942/gfx950. On RDNA,
WMMA accumulators use ordinary ArchVGPRs and the target may run wave32 or wave64;
query the compiled wave mode and re-budget its fragment layout.

- **MFMA accumulators are register-resident.** A large output tile keeps its
  whole C/D accumulator live across the K-loop. For `v_mfma_f32_16x16x16_f16`
  the accumulator is **4 VGPRs per lane (C=4, D=4)**; a 256×128 macro-tile built
  from many such MFMAs can pin a hundred-plus accumulator registers per wave.
  Accumulators are conventionally placed in **AGPRs** to free ArchVGPRs, but a
  big tile still adds to the same combined physical budget. See
  [MFMA](../hardware/mfma.md).
- **wave64.** CDNA is wave64-only, so per-lane register costs are paid across
  64 lanes; there is no wave32 fallback to halve the live state (unlike RDNA4
  gfx1201, which can run wave32).
- **Deep software pipelines.** Double-buffering and `num_stages>1` keep multiple
  tiles of A/B in registers (or in flight) at once, multiplying live state.
- **Address/index bloat.** 64-bit flat addresses, per-lane offsets, and loop
  induction variables all consume ArchVGPRs that compete with the accumulator.

## Fixes (ordered by usual payoff)

### 1. Cap the per-wave VGPR budget to hit a target occupancy

Set an explicit occupancy target and let the compiler trade tile size /
rematerialization against it. The portable HIP attribute is
`__launch_bounds__(maxThreadsPerBlock, minWavesPerEU)`:

```cpp
// gfx942/gfx950: 256 threads/block = 4 wave64 waves/block.
// minWavesPerEU=2 asks the backend to keep the combined regular+accumulator
// allocation <= 512/2 = 256 entries so >=2 waves co-reside per SIMD (EU).
__global__ void __launch_bounds__(256, 2)
hgemm_tile(const half* __restrict__ A,
           const half* __restrict__ B,
           float* __restrict__ C, int M, int N, int K) {
    // ... MFMA macro-tile loop ...
}
```

The equivalent backend knobs: `-mllvm -amdgpu-waves-per-eu=2` (Clang/HIP) and,
in Triton for AMD, `waves_per_eu=N`. See
[occupancy tuning](../techniques/occupancy-tuning.md). Pushing waves/EU up is
only a win until it forces spills — measure both occupancy *and* scratch.

### 2. Shrink the accumulator footprint

The accumulator tile is usually the single biggest consumer. Reduce the
**per-wave output tile** (e.g. 128×128 → 128×64), or split the K dimension so
each wave accumulates a shorter partial product:
[split-K](../techniques/split-k.md) and [stream-K](../techniques/stream-k.md)
both reduce the live accumulator (at the cost of a reduction epilogue). Choosing
`32x32x*` vs `16x16x*` MFMA shapes also changes the accumulator-to-issue ratio.

### 3. Keep streaming data out of VGPRs entirely

Use **direct-to-LDS** loads (`buffer_load ... lds` / `global_load_lds_*`) so
HBM→LDS transfers **bypass the VGPR file**, eliminating the staging registers a
normal load-then-`ds_write` would need. This is AMD's analog of NVIDIA
`cp.async`. See [async copy to LDS](../hardware/async-copy-lds.md).

### 4. On CDNA, use the AGPR view without mistaking remapping for a spill

Steering the accumulator into **AGPRs** frees ArchVGPRs for addressing and lets
the matrix unit co-issue with VALU. The compiler does this by default for MFMA
output. It does **not** create another occupancy pool: regular and accumulator
allocations still sum against 512. Likewise, mapping a regular live range into
an unused AGPR index is register remapping within the same file, not a scratch
spill. Tile shape and target-specific VGPR/AGPR controls can still help with
namespace pressure — see [VGPR budgeting](../techniques/vgpr-budgeting.md).

### 5. Reduce live ranges

Recompute cheap values instead of holding them (rematerialization), narrow
index types, hoist invariants out only when it doesn't extend a live range, and
prefer `__restrict__` so the compiler can reuse registers across aliasing-free
loads.

## Worked example: reading the trade-off

Suppose a gfx942 HIP GEMM reports **104 regular VGPRs + 64 AGPRs/wave** and
`ScratchSize=0`. Both counts are already multiples of eight, so the combined
allocation is 168 and `floor(512/168)=3` waves/SIMD. Two paths:

- **Down to ≤128 combined registers** (smaller tile or a higher waves/EU target):
  `floor(512/128)=4` waves/SIMD, which adds one resident wave as long as no
  scratch spills appear.
- **Grow to exactly 256 combined registers:** two waves/SIMD still fit. A legal
  allocation above 256 drops to one wave/SIMD but does **not** inherently spill;
  spill evidence comes from scratch allocation/instructions, not this threshold.

The sweet spot is the smallest tile that keeps the MFMA pipeline fed while
holding scratch at zero. Always co-optimize with LDS, since LDS per-CU
(64 kB on gfx942, 160 kB on gfx950) is the *other* occupancy limiter.

## Anti-patterns

- **Chasing maximum occupancy blindly.** A compute-bound MFMA kernel can be
  fastest at 2–4 waves/SIMD (8–16 waves/CU) if the tile is large and the
  pipeline is full; forcing 8 waves/SIMD by shrinking the tile can drop MFMA
  utilization. Tune for throughput, not the occupancy number.
- **Ignoring `ScratchSize`.** Spills can hide behind "it still runs." Always
  check the resource report.
- **Hardcoding `warpSize`/lane counts** when computing register tiling — query
  it. gfx9 CDNA is wave64, while RDNA supports target/mode-dependent wave32 or
  wave64 (see [HIP HW model](../../sources/docs/doc-rocm-hip-hw.md)).

## See also

- [Low occupancy pattern](low-occupancy.md)
- [VGPR budgeting technique](../techniques/vgpr-budgeting.md)
- [Occupancy tuning technique](../techniques/occupancy-tuning.md)
- [Wavefront, registers & occupancy](../hardware/wavefront.md)
- [MFMA matrix cores](../hardware/mfma.md)

## Sources

- [AMD HIP — Hardware Implementation / Performance Guidelines](https://rocm.docs.amd.com/projects/HIP/en/latest/understand/hardware_implementation.html)
- [LLVM AMDGPU Backend User Guide (occupancy & register controls)](https://llvm.org/docs/AMDGPUUsage.html)
- [Matrix Core Programming on CDNA3 and CDNA4](https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores-cdna/README.html)
- [Optimizing GEMM on AMD GPUs (occupancy & tile sizing)](https://rocm.blogs.amd.com/artificial-intelligence/matrix-cores/README.html)
