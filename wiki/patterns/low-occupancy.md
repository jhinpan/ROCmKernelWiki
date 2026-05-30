---
id: pattern-low-occupancy
title: "Low Occupancy / Idle CUs"
type: pattern
architectures:
- gfx942
- gfx950
- gfx1201
tags:
- low-occupancy
- idle-cu
- latency-bound
- occupancy-tuning
- vgpr-budgeting
- wave64
- vgpr
- cu
- agpr
symptoms:
- low-occupancy
- idle-cu
- latency-bound
candidate_techniques:
- technique-occupancy-tuning
- technique-vgpr-budgeting
related:
- hw-wavefront
- technique-occupancy-tuning
- technique-vgpr-budgeting
- pattern-vgpr-pressure
- pattern-tail-effect
sources:
- hw-wavefront
- doc-rocm-hip-hw
- blog-triton-optimizations
- doc-cdna3-isa
- blog-gemm-optimization
---

# Low Occupancy / Idle CUs

## What this pattern looks like

A kernel runs far below the hardware's achievable throughput, and a profile
shows the compute units (CUs) are *waiting*, not *working*. Concretely you see
one or more of:

- **Low occupancy** — the SIMDs hold only a handful of resident wavefronts per
  CU, well below the architectural maximum of **40 waves/CU** (4 SIMD16 pools ×
  10 waves each on CDNA).
- **Idle CUs** — `rocprof` / `rocprofv3` reports a low `GRBM_GUI_ACTIVE` /
  valu-busy fraction, or whole CUs never receive a workgroup.
- **Latency-bound** — wave-level analysis shows long stalls on `s_waitcnt`
  (waiting for `vmcnt`/`lgkmcnt`), with the VALU and matrix cores idle in
  between. There simply aren't enough other waves resident to hide that latency.

Occupancy is the GPU's only latency-hiding mechanism: a memory load takes
hundreds of cycles, and the scheduler hides that by switching to another ready
wave. If too few waves are resident, the SIMD stalls and the CU goes idle even
though work remains.

## Why it happens

Occupancy is the *minimum* of several per-CU resource limits. The binding
constraint is usually one of these:

| Limiting resource | CDNA3 (gfx942) | CDNA4 (gfx950) | Effect |
|---|---|---|---|
| VGPRs | 256 Arch + 256 Acc / wave, allocated in groups of 8 dwords | same | High VGPR/wave → fewer waves fit |
| LDS | 64 KB/CU | 160 KB/CU | Large `__shared__` tiles cap workgroups/CU |
| Waves/CU | 40 (4×10) | 40 (4×10) | Hard architectural ceiling |
| Workgroup count | grid size | grid size | Too few blocks → idle CUs |

On CDNA, **VGPR pressure is the most common cause.** Because allocation is in
groups of 8 dwords and a CU has a fixed register file, a wave using 128 VGPRs
allows ~2× the residency of one using 256 VGPRs. Matrix-heavy kernels are often
**AGPR-bound** instead — large accumulator tiles consume the 256 AccVGPRs and
throttle residency the same way (see
[VGPR/AGPR pressure](vgpr-pressure.md)).

A second, distinct cause is simply **not enough workgroups**: if the grid
launches fewer blocks than there are CUs (304 on MI300X, 256 on MI355X), some
CUs never get work. The related [tail effect](tail-effect.md) is the same
problem at the *end* of a wave of blocks.

## Diagnosis

Start with the static register/LDS budget the compiler chose, then confirm with
a runtime profile.

```bash
# 1) Static: what did the compiler allocate? Look for .vgpr_count,
#    .agpr_count, .sgpr_count, .lds_size in the kernel metadata.
hipcc --offload-arch=gfx942 -Rpass-analysis=kernel-resource-usage \
      -c my_kernel.hip -o my_kernel.o

# 2) Runtime: occupancy + stall reasons (rocprofiler-sdk / rocprofv3)
rocprofv3 --pmc GRBM_GUI_ACTIVE SQ_WAVES SQ_BUSY_CYCLES \
          SQ_WAIT_INST_LDS SQ_INSTS_VALU -- ./my_app
```

A useful first sanity check from host code is the occupancy API, which reports
the max resident blocks given the kernel's actual resource use:

```cpp
#include <hip/hip_runtime.h>

int max_blocks = 0;
constexpr int kBlockSize = 256;        // 256 threads = 4 wave64 wavefronts
size_t dyn_lds = 0;

hipOccupancyMaxActiveBlocksPerMultiprocessor(
    &max_blocks, (const void*)my_kernel, kBlockSize, dyn_lds);

// waves resident per CU at full occupancy:
//   waves = max_blocks * (kBlockSize / warpSize)   // warpSize == 64 on gfx9
// If this is << 40, you are occupancy-limited; find the binding resource.
printf("max blocks/CU = %d  -> ~%d waves/CU (ceiling 40)\n",
       max_blocks, max_blocks * (kBlockSize / 64));
```

> `warpSize` is **64 on gfx9 (CDNA)** and 32 on gfx10+/RDNA — query it, never
> hardcode. A 256-thread block is 4 wave64 wavefronts on MI300, but 8 wave32
> wavefronts on a wave32 RDNA4 launch.

## Fixes

The two primary levers are captured as techniques:

1. **[Occupancy tuning](../techniques/occupancy-tuning.md)
   (`technique-occupancy-tuning`).** Raise residency by reducing the binding
   resource: shrink the LDS tile, pick a smaller block size, or cap VGPRs with
   launch bounds / scheduler hints. In HIP, `__launch_bounds__(maxThreads,
   minWavesPerEU)` tells the compiler to limit register allocation so the
   requested number of waves fits.

   ```cpp
   // Ask the compiler to keep enough VGPR headroom for >=8 waves/SIMD.
   // It will spill or rematerialize rather than exceed the budget.
   __global__ void __launch_bounds__(256, 2) my_kernel(/* ... */) {
       // ... two workgroups (8 wave64 waves) target per CU ...
   }
   ```

   In the [Triton AMD backend](../languages/triton-amd.md) the equivalent knob
   is `waves_per_eu` (plus `num_warps` / `num_stages`); raising it pushes the
   compiler toward lower register use and higher occupancy. See the
   [Triton optimization guide](blog-triton-optimizations).

2. **[VGPR budgeting](../techniques/vgpr-budgeting.md)
   (`technique-vgpr-budgeting`).** Reduce live registers structurally: shorten
   live ranges, reuse registers, move accumulators to AGPRs to free ArchVGPRs,
   use `buffer_load ... lds` / `global_load_lds` (direct-to-LDS) to stream HBM
   into LDS **without** transiting VGPRs, and avoid large unrolled prologues
   that inflate the live set. Crossing a group-of-8 boundary (e.g. 168 → 160
   VGPRs) can unlock a whole extra wave per SIMD.

If the grid is too small, the fix is *more* parallelism rather than *less*
resource use: increase the block count (split a long K loop with
[split-K](../techniques/split-k.md) or
[stream-K](../techniques/stream-k.md)), or use a
[persistent kernel](../techniques/persistent-kernel.md) so a fixed pool of
workgroups iterates over tiles and keeps every CU busy.

## The occupancy trade-off — don't over-correct

Higher occupancy is **not** always faster. Latency-bound kernels benefit, but
compute-bound GEMM-style kernels often run *fastest* at modest occupancy (2–4
waves/CU) because large register/LDS tiles maximize data reuse and matrix-core
utilization. Pushing occupancy up by shrinking tiles can trade away that reuse
and regress performance. The goal is "enough waves to hide the stalls you
actually have," not "maximum waves." Measure stall reasons first
([latency-bound vs compute-bound](memory-bound.md)) before deciding which way to
move.

A practical loop:

1. Profile → confirm CUs are *stalled on `s_waitcnt`*, not compute-saturated.
2. Find the binding resource (VGPR/AGPR, LDS, or grid size).
3. Relieve exactly that one, re-measure, and stop when the stalls disappear or
   reuse starts to suffer.

## See also

- [Wavefront, registers & occupancy](../hardware/wavefront.md)
- [VGPR/AGPR pressure pattern](vgpr-pressure.md)
- [Tail effect pattern](tail-effect.md)
- [Occupancy tuning](../techniques/occupancy-tuning.md) ·
  [VGPR budgeting](../techniques/vgpr-budgeting.md)

## Sources

- [HIP Programming Guide — Hardware Features & Occupancy](https://rocm.docs.amd.com/projects/HIP/en/latest/)
- [AMD CDNA3 ISA Reference Guide](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf)
- [Triton kernel optimization on AMD GPUs](https://rocm.blogs.amd.com/)
- [GEMM optimization on AMD GPUs](https://rocm.blogs.amd.com/)
