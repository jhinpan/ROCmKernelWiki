---
id: technique-occupancy-tuning
title: Occupancy Tuning — Waves per SIMD vs ILP on CDNA
type: technique
version_sensitive:
- vs-wave-slots-cdna3-cdna4
- vs-cdna-unified-vgpr-agpr-allocation
architectures:
- gfx942
- gfx950
tags:
- occupancy-tuning
- vgpr
- lds
- agpr
- wave64
- vgpr-budgeting
- software-pipelining
confidence: source-reported
reproducibility: snippet
hardware_features:
- vgpr
- agpr
- lds
- wave64
languages:
- hip
- triton
related:
- hw-wavefront
- technique-vgpr-budgeting
- technique-lds-double-buffering
- technique-mfma-pipelining
- pattern-low-occupancy
- pattern-vgpr-pressure
sources:
- doc-rocm-hip-hw
- blog-triton-optimizations
- blog-gemm-optimization
- doc-cdna3-isa
- blog-4wave-fp8-gemm
- blog-amdgpu-kernel-opt-guide
implemented_by:
- pr-Tensile-1529
- pr-FlyDSL-591
- pr-composable_kernel-2825
- pr-composable_kernel-2722
- pr-Tensile-1406
- pr-FlyDSL-447
---
# Occupancy Tuning — Waves per SIMD vs ILP on CDNA

## What occupancy actually is on CDNA

A CDNA Compute Unit (CU) is built from **4 SIMD16 units** (the "execution units",
or EUs). Each SIMD time-slices a pool of resident **wave64** wavefronts, switching
between them to hide memory and instruction latency. The CDNA3/CDNA4 ceiling is
**8 waves per SIMD → 32 waves per CU** (4 pools × 8). *Occupancy* is the number
of resident waves you actually achieve, usually expressed as waves/SIMD or as a
percentage of that 32-wave ceiling.

You almost never hit 32. Keep all terms in the same unit when estimating the
minimum. Compute register capacity per SIMD, then convert it to waves/CU before
comparing it with the LDS limit:

```
# HSA metadata .vgpr_count is already combined on gfx942 and gfx950.
vector_alloc = round_up(metadata_vgpr_count, 8)
sgpr_alloc = round_up(sgpr_count_including_target_specials, 16)

vector_limited_waves_per_simd = min(8, floor(512 / vector_alloc))
vector_limited_waves_per_cu = 4 * vector_limited_waves_per_simd
sgpr_limited_waves_per_simd = min(
    8, floor(sgpr_capacity_per_simd / sgpr_alloc))
sgpr_limited_waves_per_cu = 4 * sgpr_limited_waves_per_simd

lds_limited_waves_per_cu =
    floor(lds_per_cu / allocated_lds_per_workgroup) * waves_per_workgroup

resident_waves_per_cu = min(
    32,
    vector_limited_waves_per_cu,
    sgpr_limited_waves_per_cu,
    lds_limited_waves_per_cu,
    other_workgroup_and_scheduler_limits)
```

The two knobs you almost always end up fighting are **VGPR count** and **LDS
bytes per workgroup**. Both are reported by the compiler at build time, so you can
predict occupancy before you ever run the kernel.

## Reading your resource usage

Ask the compiler. With `-Rpass-analysis=kernel-resource-usage` (or just
`--save-temps` and grep the `.s`), Clang prints the VGPR/AGPR/SGPR/LDS footprint
per kernel:

```bash
hipcc -O3 --offload-arch=gfx942 \
      -Rpass-analysis=kernel-resource-usage \
      -c gemm.hip -o gemm.o
# ... remark: NumVgprs: 168   NumAgprs: 64   ScratchSize: 0
#     LDSSize: 32768   ...
```

`rocprofv3`/`rocprof` will also report achieved `VALUUtilization` and an occupancy
estimate per dispatch — compare the static prediction against the dynamic number
to catch scratch spills.

For code-object/HSA metadata on either target, round the combined `.vgpr_count`
to eight and do not add `.agpr_count` again. If a lower-level compiler remark
instead exposes separate `NumVgprs` and `NumAgprs`, use the target-specific
derivation documented on the [wavefront page](../hardware/wavefront.md);
`TotalNumVgprs`, when present, is already combined.

## VGPR-limited occupancy

VGPRs are allocated per wave in **groups of 8 dwords**, and a wave can hold up to
512 total (up to 256 regular ArchVGPR names plus up to 256 AccVGPR/AGPR names).
CDNA3/CDNA4 use one combined 512-entry-per-lane capacity per SIMD, so regular
and accumulator allocations are **summed**, not treated as independent banks.
Because that capacity is fixed, the number of waves that fit scales inversely
with the combined per-wave allocation: rounding usage down across a residency
boundary lets one more wave move in.
As a rule of thumb on gfx942, *halving* a kernel's VGPR footprint roughly *doubles*
its VGPR-limited occupancy until you saturate the 8-wave slot cap.

MFMA accumulators use the **AGPR register view** (see
[wavefront / register files](../hardware/wavefront.md)), but their allocation
consumes the same combined physical capacity. A large `v_mfma`-tiled GEMM can
therefore be accumulator-dominated even when its regular addressing code is
lean. Shrinking the macro-tile (fewer accumulators) is the lever there — see
[VGPR budgeting](vgpr-budgeting.md).

If the compiler cannot fit your VGPRs at the requested occupancy it **spills to
scratch** (`ScratchSize > 0`), which is global-memory traffic on the hot path —
almost always worse than simply accepting lower occupancy. Investigate any
hot-path scratch and distinguish compiler spills from explicit private objects.

## LDS-limited occupancy

LDS is a **per-CU** resource shared by all resident workgroups: **64 kB/CU on
gfx942**, **160 kB/CU on gfx950**. The number of concurrent workgroups is

```
workgroups_per_cu = floor(LDS_per_cu / allocated_LDS_bytes_per_workgroup)
```

Example (gfx942): a workgroup of 256 threads (= 4 wave64s) that statically
allocates a 32 kB `__shared__` double-buffer fits only `floor(65536 / 32768) = 2`
workgroups per CU → **8 waves/CU**, i.e. 25% of the 32-wave ceiling, *before*
VGPRs are even considered. On gfx950, LDS is allocated in 1280-byte units, so
32 KiB rounds up to `round_up(32768, 1280) = 33280` bytes. The 160 KiB LDS then
fits `floor(163840 / 33280) = 4` workgroups → **16 waves/CU** (not five).
This is exactly why
[LDS double-buffering](lds-double-buffering.md) trades occupancy for latency
hiding — and why the larger gfx950 LDS relaxes that trade.

## Forcing a target with `__launch_bounds__`

HIP lets you cap block size and request a **minimum** waves-per-EU; the compiler
then bounds register allocation to honor it (potentially spilling if impossible):

```cpp
// 256 threads/block (4 wave64s), ask the allocator to keep >=2 waves/EU resident.
// Lower waves_per_eu  -> allocator may use MORE VGPRs (better ILP, fewer waves)
// Higher waves_per_eu -> allocator caps VGPRs (more TLP, risk of spills)
__global__ void __launch_bounds__(256, /*min_waves_per_eu=*/2)
gemm_tile(const float* __restrict__ A,
          const float* __restrict__ B,
          float* __restrict__ C, int M, int N, int K)
{
    // ... MFMA macro-tile; accumulators in AGPRs ...
}
```

Clang also exposes the finer-grained
`__attribute__((amdgpu_waves_per_eu(min, max)))`. Setting `(min,min)` is the
common idiom to pin a kernel to a specific occupancy point you found by sweeping.

## The occupancy ↔ ILP trade-off

More waves is **not** automatically faster. Two latency-hiding strategies compete
for the same register file:

- **TLP (high occupancy):** many small waves; the SIMD hides a stalled wave by
  switching to another. Wins for **memory-bound, low-arithmetic** kernels
  (elementwise, reductions, attention softmax tails) where there is little to do
  per byte and you just need enough waves to cover HBM latency.
- **ILP (low occupancy, fat registers):** few waves, each holding a large
  register tile so independent FMAs/MFMAs are in flight simultaneously and data is
  reused from registers instead of re-fetched. Wins for **compute-bound GEMM /
  conv**, where a big accumulator tile maximizes matrix-core utilization.

Well-tuned CDNA GEMMs frequently run **best at only 1–2 waves/SIMD**: the large
MFMA accumulator tile that creates the ILP also burns the AGPRs that would
otherwise raise occupancy. Pushing occupancy up there *shrinks the tile* and
*lowers* throughput. The
[ping-pong / 4-wave FP8 GEMM schedule](../kernels/fp8-gemm.md) is a deliberate
low-occupancy design that pairs a small number of waves with software pipelining
to keep the matrix cores saturated.

A practical heuristic: find the *minimum* occupancy that fully hides your dominant
latency (HBM for memory-bound, MFMA issue for compute-bound), then spend every
remaining register on tile size / pipelining. Don't maximize occupancy as a goal
in itself.

## Tuning from Triton

The Triton AMD backend exposes occupancy directly as the **`waves_per_eu`** knob
(alongside `matrix_instr_nonkdim`, `num_stages`, and `kpack`). It is the same
compiler hint as the HIP attribute — a *target*, not a guarantee:

```python
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 256, 'BLOCK_K': 64},
                      num_warps=8, num_stages=2,
                      kwargs={'waves_per_eu': 0}),   # 0 = let backend choose
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 64},
                      num_warps=4, num_stages=2,
                      kwargs={'waves_per_eu': 2}),    # push occupancy up
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64},
                      num_warps=8, num_stages=3,
                      kwargs={'waves_per_eu': 1}),    # fat tile, low occupancy
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def gemm_kernel(a_ptr, b_ptr, c_ptr, M, N, K, waves_per_eu: tl.constexpr,
                BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    ...  # tl.dot(...) lowers to v_mfma_*
```

In practice you autotune `waves_per_eu` *jointly* with the block shape and
`num_stages`, because a higher occupancy target tightens the register budget
while a deeper pipeline consumes it; their combination can trip a scratch
spill. AMD's Triton optimization guide
recommends sweeping `waves_per_eu ∈ {0,1,2,3,4}` against your real problem sizes
rather than guessing.

## Checklist

1. Build with `-Rpass-analysis=kernel-resource-usage`; record VGPR/AGPR/LDS.
2. Confirm **ScratchSize == 0** — fix spills before chasing occupancy.
3. Compute the static `waves_per_simd` ceiling from each resource; identify the
   binding one (combined regular-plus-accumulator vector allocation, or LDS).
4. Classify the kernel: memory-bound → raise occupancy; compute-bound → favor ILP
   / bigger tile at low occupancy.
5. Sweep `__launch_bounds__` / `waves_per_eu` and measure; keep the fastest point,
   not the highest-occupancy point.

> **Concrete CDNA3/CDNA4 limits and guide corrections:** the general scalar
> namespace is `s0`–`s101` (102 names). `.sgpr_count` is a per-wave raw count
> that also includes enabled target-special pairs such as VCC, FLAT_SCRATCH, and
> XNACK; GFX9 rounds it to 16-register allocation blocks, so the encoded
> allocation can reach 112. AMDHSA **User SGPR** instead means at most 16
> dispatch-initialized registers. The guide's 104-SGPR per-workgroup wording
> conflates these concepts. A wave can name up to **256 regular VGPRs**
> and **256 accumulator VGPRs**; their target-specific combined allocation is
> encoded in groups of 8 against one 512-entry-per-lane SIMD capacity. Exactly
> 256 combined entries permit two
> waves/SIMD; a legal allocation above 256 permits one and does not inherently
> spill. The guide's “default 128” cap is compiler/snapshot-specific rather than
> an architectural limit. Remapping a regular value to an unused AGPR index is
> register allocation inside that unified file, not a spill; actual spills are
> identified by scratch/private-segment allocation and scratch instructions.

## See also

- [Wavefront & register files](../hardware/wavefront.md)
- [VGPR budgeting](vgpr-budgeting.md)
- [LDS double-buffering](lds-double-buffering.md)
- [Pattern: low occupancy](../patterns/low-occupancy.md)
- [Pattern: VGPR pressure](../patterns/vgpr-pressure.md)

## Sources

- [HIP Performance Guidelines (occupancy, launch bounds)](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/performance_guidelines.html)
- [Triton kernel performance optimization on AMD](https://rocm.blogs.amd.com/software-tools-optimization/triton-kernel-optimization/README.html)
- [GEMM kernel optimization on AMD GPUs](https://rocm.blogs.amd.com/artificial-intelligence/matrix-cores/README.html)
- [AMDGPU Kernel Optimization Guide (captured snapshot)](https://github.com/nod-ai/amd-shark-ai/blob/efa471aeef66a260c85983cc41e833bfa769dade/docs/amdgpu_kernel_optimization_guide.md) — register claims, launch bounds, and documented corrections
- [AMD Instinct MI300 / CDNA3 ISA Reference Guide](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf)
- [Optimizing an FP8 GEMM with a 4-wave ping-pong schedule](https://rocm.blogs.amd.com/artificial-intelligence/fp8-gemm/README.html)
