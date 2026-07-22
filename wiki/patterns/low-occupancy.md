---
id: pattern-low-occupancy
title: Low Occupancy / Idle CUs
type: pattern
version_sensitive:
- vs-wave-slots-cdna3-cdna4
- vs-cdna-unified-vgpr-agpr-allocation
architectures:
- gfx942
- gfx950
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
implemented_by:
- pr-Tensile-1383
---
# Low Occupancy / Idle CUs

## What this pattern looks like

A kernel runs far below the hardware's achievable throughput, and a profile
shows the compute units (CUs) are *waiting*, not *working*. Concretely you see
one or more of:

- **Low occupancy** — each SIMD holds only a handful of resident wavefronts,
  well below the CDNA3/CDNA4 architectural maximum of **8 waves/SIMD** or
  **32 waves/CU** (4 SIMD16 pools × 8 waves each).
- **Idle CUs** — `rocprof` / `rocprofv3` reports a low `GRBM_GUI_ACTIVE` /
  valu-busy fraction, or whole CUs never receive a workgroup.
- **Latency-bound** — wave-level analysis shows long stalls on `s_waitcnt`
  (waiting for `vmcnt`/`lgkmcnt`), with the VALU and matrix cores idle in
  between. Too few ready resident waves may be contributing to that latency.

Occupancy is one important latency-hiding mechanism: the scheduler can switch
from a stalled wave to another ready resident wave. It is not the only one;
instruction-level parallelism, multiple outstanding memory operations, software
pipelining, asynchronous/direct-to-LDS transfers, and cache reuse can hide or
avoid latency within a small set of waves. Low occupancy becomes a problem when
those waves also lack enough independent ready work.

## Why it happens

Occupancy is the *minimum* of several resource and placement limits. Convert
per-SIMD register ceilings to waves/CU before comparing them with per-CU LDS or
grid limits. The binding constraint is usually one of these:

| Limiting resource | CDNA3 (gfx942) | CDNA4 (gfx950) | Effect |
|---|---|---|---|
| Vector registers | One combined 512-entry-per-lane SIMD budget | HSA metadata `.vgpr_count`, rounded to 8 | High combined allocation/wave → fewer waves fit |
| Scalar registers | Raw per-wave `.sgpr_count`, then allocated in groups of 16 | same | High scalar allocation can also cap waves/SIMD |
| LDS | 64 KB/CU | 160 KB/CU | Large `__shared__` tiles cap workgroups/CU |
| Waves/CU | 32 (4×8) | 32 (4×8) | Hard architectural ceiling |
| Workgroup count | grid size | grid size | Too few blocks → idle CUs |

On CDNA, **VGPR pressure is the most common cause.** Because allocation is in
groups of 8 dwords and each SIMD has a fixed 512-entry-per-lane vector capacity,
a wave with 128 combined regular-plus-accumulator registers allows twice the
vector-limited residency of one with 256 (4 versus 2 waves/SIMD). Matrix-heavy
kernels are often **accumulator-dominated** — large AGPR tiles add to that same
combined allocation (see
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

On gfx942 and gfx950, HSA metadata `.vgpr_count` already includes the
accumulator subset, so round it to eight and do not add `.agpr_count`. If you
are reading separate lower-level `NumVgprs`/`NumAgprs` compiler remarks instead
of code-object metadata, use the derivation on the
[wavefront page](../hardware/wavefront.md).

A useful first sanity check from host code is the occupancy API, which reports
the max resident blocks given the kernel's actual resource use:

```cpp
#include <hip/hip_runtime.h>

int max_blocks = 0;
constexpr int kBlockSize = 256;
size_t dyn_lds = 0;
int dev = 0;
hipDeviceProp_t props{};
hipGetDeviceProperties(&props, dev);

hipOccupancyMaxActiveBlocksPerMultiprocessor(
    &max_blocks, (const void*)my_kernel, kBlockSize, dyn_lds);

int waves_per_block = (kBlockSize + props.warpSize - 1) / props.warpSize;
int waves_per_cu = max_blocks * waves_per_block;
int wave_ceiling = props.maxThreadsPerMultiProcessor / props.warpSize;
printf("max blocks/CU = %d -> ~%d waves/CU (device ceiling %d)\n",
       max_blocks, waves_per_cu, wave_ceiling);
```

> `warpSize` is **64 on gfx9 (CDNA)**. RDNA can be compiled in wave32 or wave64
> mode, so query it rather than inferring it from the gfx generation. A
> 256-thread block is four wave64 wavefronts or eight wave32 wavefronts.

## Fixes

The two primary levers are captured as techniques:

1. **[Occupancy tuning](../techniques/occupancy-tuning.md)
   (`technique-occupancy-tuning`).** Raise residency by reducing the binding
   resource: shrink the LDS tile, pick a smaller block size, or cap VGPRs with
   launch bounds / scheduler hints. In HIP, `__launch_bounds__(maxThreads,
   minWavesPerEU)` tells the compiler to limit register allocation so the
   requested number of waves fits.

   ```cpp
   // gfx942/gfx950: ask for enough register headroom for >=2 waves/SIMD.
   // The combined vector-register target is <= 512/2 = 256 entries/wave.
   __global__ void __launch_bounds__(256, 2) my_kernel(/* ... */) {
       // ... kernel body ...
   }
   ```

   In the [Triton AMD backend](../languages/triton-amd.md) the equivalent knob
   is `waves_per_eu` (plus `num_warps` / `num_stages`); raising it pushes the
   compiler toward lower register use and higher occupancy. See the
   [Triton optimization guide](../../sources/blogs/blog-triton-optimizations.md).

2. **[VGPR budgeting](../techniques/vgpr-budgeting.md)
   (`technique-vgpr-budgeting`).** Reduce live registers structurally: shorten
   live ranges and reuse regular names. On CDNA MFMA kernels, use the AGPR view
   for accumulators without treating it as a separate occupancy pool; RDNA has
   no AGPR namespace, so budget its WMMA fragments as ArchVGPRs. Also
   use `buffer_load ... lds` / `global_load_lds` (direct-to-LDS) to stream HBM
   into LDS **without** transiting VGPRs, and avoid large unrolled prologues
   that inflate the live set. On gfx942/gfx950, crossing a residency boundary (for example,
   reducing the rounded combined allocation from 136 to 128) can raise the
   vector limit from 3 to 4 waves/SIMD.

If the grid is too small, the fix is *more* parallelism rather than *less*
resource use: increase the block count (split a long K loop with
[split-K](../techniques/split-k.md) or
[stream-K](../techniques/stream-k.md)), or use a
[persistent kernel](../techniques/persistent-kernel.md) so a fixed pool of
workgroups iterates over tiles and keeps every CU busy.

## The occupancy trade-off — don't over-correct

Higher occupancy is **not** always faster. Latency-bound kernels benefit, but
compute-bound GEMM-style kernels often run *fastest* at modest occupancy (for
example, 2–4 waves/SIMD or 8–16 waves/CU on gfx942/gfx950) because large register/LDS tiles maximize data
reuse and matrix-core utilization. Pushing occupancy up by shrinking tiles can
trade away that reuse and regress performance. The goal is "enough ready work
to hide the stalls you actually have," not "maximum waves." Measure stall
reasons first
([latency-bound vs compute-bound](memory-bound.md)) before deciding which way to
move.

A practical loop:

1. Profile → confirm CUs are *stalled on `s_waitcnt`*, not compute-saturated.
2. Find the binding resource (combined vector allocation, LDS, or grid size).
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
