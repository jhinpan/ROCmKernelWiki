---
id: pattern-vgpr-pressure
title: VGPR Pressure, Register Spills, and Occupancy Collapse
type: pattern
architectures:
- gfx942
- gfx950
- gfx1201
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
shows few waves resident per CU, and latency from a single memory miss is never
hidden because there are no other waves to switch to. The root cause is almost
always that **each wave consumes too many vector registers**, so the CU cannot
keep enough waves in flight to cover memory and MFMA latency. In the worst case
the compiler runs out of registers entirely and inserts **spill/reload traffic
to scratch** (private memory backed by HBM), which is both slow and consumes
even more VMEM bandwidth.

On CDNA the VGPR file is the scarcest occupancy resource for most compute-bound
kernels. A wave can hold up to **512 VGPRs (256 ArchVGPRs + 256 AccVGPRs/AGPRs)**,
allocated in **groups of 8 dwords**, and the CU supports **up to 40 waves
(4 SIMD pools × 10 waves)**. Because the register file is fixed, occupancy is
roughly `floor(file_per_SIMD / per_wave_VGPRs)` capped at 10 per SIMD — so VGPR
count per wave directly throttles how many waves fit. See
[wavefront & occupancy](../hardware/wavefront.md).

## Symptoms and how to confirm them

| Symptom | Where you see it | What it means |
|---|---|---|
| `vgpr-pressure` | High `VGPRs` in the compiler resource report / ISA `.vgpr_count` | Per-wave allocation is large, capping occupancy |
| `register-spill` | Non-zero `ScratchSize` / `.private_segment_fixed_size`; `scratch_load/store` in ISA | Compiler ran out of registers; spilling to HBM-backed scratch |
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

Any non-zero `.private_segment_fixed_size` (scratch) on a hot kernel is a red
flag: the compiler spilled. The first goal is almost always to get spills to
**zero**, then to push occupancy.

## Why it happens on AMD specifically

- **MFMA accumulators are register-resident.** A large output tile keeps its
  whole C/D accumulator live across the K-loop. For `v_mfma_f32_16x16x16_f16`
  the accumulator is **4 VGPRs per lane (C=4, D=4)**; a 256×128 macro-tile built
  from many such MFMAs can pin a hundred-plus accumulator registers per wave.
  Accumulators are conventionally placed in **AGPRs** to free ArchVGPRs, but a
  big tile is still ultimately AGPR-bound. See [MFMA](../hardware/mfma.md).
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
// 256 threads/block = 4 wave64 waves/block.
// minWavesPerEU=2 tells the backend to keep <= ~256/(2*?) VGPRs so >=2
// waves co-reside per SIMD (EU). Raising it forces a smaller VGPR budget.
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

### 4. Move accumulators into AGPRs / rebalance the two banks

Steering the accumulator into **AGPRs** frees ArchVGPRs for addressing and lets
the matrix unit co-issue with VALU. The compiler does this by default for MFMA
output, but tile shape and `-amdgpu-num-vgpr`/agpr controls let you rebalance
the 256+256 split — see [VGPR budgeting](../techniques/vgpr-budgeting.md).

### 5. Reduce live ranges

Recompute cheap values instead of holding them (rematerialization), narrow
index types, hoist invariants out only when it doesn't extend a live range, and
prefer `__restrict__` so the compiler can reuse registers across aliasing-free
loads.

## Worked example: reading the trade-off

Suppose a HIP GEMM reports **168 VGPRs/wave** and `ScratchSize=0`. With 256
ArchVGPRs available per SIMD-lane budget, `floor(256/168)=1` wave/SIMD → very
low occupancy but no spills. Two paths:

- **Down to ≤128 VGPRs** (smaller tile or `__launch_bounds__(…, 2)`):
  `floor(256/128)=2` waves/SIMD — occupancy doubles, latency hiding improves,
  *as long as no spills appear*.
- **Force ≤85 VGPRs** to chase 3 waves/SIMD: if the accumulator no longer fits,
  the compiler spills (`ScratchSize>0`) and you trade a register stall for an
  HBM round-trip — usually a net loss.

The sweet spot is the smallest tile that keeps the MFMA pipeline fed while
holding scratch at zero. Always co-optimize with LDS, since LDS per-CU
(64 kB on gfx942, 160 kB on gfx950) is the *other* occupancy limiter.

## Anti-patterns

- **Chasing maximum occupancy blindly.** A compute-bound MFMA kernel can be
  fastest at 2–4 waves/CU if the tile is large and the pipeline is full; forcing
  8 waves by shrinking the tile can drop MFMA utilization. Tune for throughput,
  not the occupancy number.
- **Ignoring `ScratchSize`.** Spills can hide behind "it still runs." Always
  check the resource report.
- **Hardcoding `warpSize`/lane counts** when computing register-tiling — query
  it; it is 64 on gfx9 and 32 on gfx10+ (see [HIP HW model](../../sources/docs/doc-rocm-hip-hw.md)).

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
