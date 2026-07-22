---
id: hw-wavefront
title: Wavefronts, EXEC Mask & Register Files (CDNA)
type: hardware
version_sensitive:
- vs-wave-slots-cdna3-cdna4
- vs-cdna-unified-vgpr-agpr-allocation
architectures:
- gfx942
- gfx950
tags:
- wave64
- sgpr
- vgpr
- agpr
- cu
- occupancy-tuning
confidence: source-reported
related:
- hw-mfma
- hw-cross-lane
- technique-occupancy-tuning
- technique-vgpr-budgeting
- pattern-low-occupancy
- lang-hip
sources:
- doc-cdna3-isa
- doc-rocm-hip-hw
- doc-cdna3-whitepaper
- doc-llvm-amdgpu
- blog-gemm-optimization
- blog-amdgpu-kernel-opt-guide
aliases:
- wavefront
- wave64
- warp
- subgroup
- workgroup
- thread block
- CTA
- SIMD lane
- EXEC mask
- occupancy
---
# Wavefronts, EXEC Mask & Register Files (CDNA)

## Overview

A **wavefront** (AMD's "warp") is the unit of SIMD execution on a CDNA Compute
Unit (CU). On CDNA (gfx9xx, including gfx942/CDNA3 and gfx950/CDNA4) a wavefront
is **always 64 work-items wide — wave64**. The kernel-visible width is
reported by `warpSize`; query it rather than carrying CUDA's 32-lane assumption.

All 64 lanes of a wavefront share one program counter and step through the same
instruction stream in lockstep. Per-lane divergence is handled by masking, not by
independent PCs (there is no per-lane PC / independent thread scheduling like
NVIDIA Volta+). The active-lane set is tracked by the **EXEC mask**.

A CU is partitioned into **4 SIMD units (SIMD16)**. Each SIMD owns a slice of the
register file and a pool of wave slots; the four pools together hold up to
**32 wavefronts per CU (4 × 8)**. The number actually resident — the
**occupancy** — is almost always limited by VGPR and LDS usage rather than by the
32-wave hardware ceiling.

> Verified on an AMD Instinct MI350X (gfx950, ROCm 7.2): `rocminfo` reports
> `Max Waves Per CU: 32` and `Max Work-item Per CU: 2048` (= 32 × wave64), and
> `hipGetDeviceProperties` returns `maxThreadsPerMultiProcessor = 2048`. gfx942
> (CDNA3) likewise caps at 32 waves/CU. The 40-wave/4×10 figure quoted for some
> older AMD GPUs is a pre-CDNA (GCN/Vega) number and does **not** apply to
> CDNA3/CDNA4.

## Terminology crosswalk

The nod-ai optimization guide uses Vulkan and AMDGPU terminology while many
kernel authors arrive from CUDA. These names describe roughly the same level of
the execution hierarchy:

| Vulkan | CUDA | AMDGPU | Normally executes on |
|---|---|---|---|
| invocation / thread | thread | work-item / thread | SIMD lane |
| subgroup | warp | wavefront | one SIMD |
| workgroup | thread block / CTA | workgroup | normally one CU |
| — | thread-block cluster | — | multiple blocks associated with a higher-level engine |
| workgroup counts | grid | grid / dispatch | GPU or logical partition |
| workgroup memory | shared memory | Local Data Share (LDS) | CU-local LDS and crossbar |

This is a vocabulary crosswalk, not a claim that the programming models are
identical. In particular, CUDA thread-block clusters expose placement and
communication semantics that cannot be inferred merely by calling an AMD
Shader Engine the corresponding hardware level.

## Workgroup placement and SIMD execution

- In the normal non-`tgsplit` mode, a workgroup is placed on one CU and is not
  migrated while it runs; its waves may occupy different SIMDs inside that CU.
  LLVM AMDGPU also documents a gfx942/gfx950 target-specific `tgsplit` mode that permits
  waves from one workgroup to be distributed across CUs. That mode does not
  allocate local memory and a shader using it cannot use the local address space
  (LDS), so it is not an alternative placement model for ordinary shared-memory
  workgroups.
- Each wave is assigned to one SIMD. On CDNA a wave64 running on SIMD16 normally
  executes a VALU operation over four 16-lane issue quarters. This VALU fact
  must not be generalized to LDS: LDS phase groups depend on the instruction
  width and architecture (see [LDS](lds.md)).
- A workgroup may contain at most 16 wave64 waves. A 256-thread workgroup makes
  four waves and is therefore a useful initial shape for exposing one wave to
  each of the four SIMDs. Placement and readiness still decide whether all four
  execute concurrently.
- The guide suggests 128 threads when reducing power matters. Treat that as a
  workload-specific heuristic rather than a general performance guarantee;
  register pressure, LDS, grid size, and latency hiding remain decisive.

## The EXEC mask and per-lane predication

`EXEC` is a 64-bit special scalar register: bit *i* enables lane *i*. A vector
(VALU) instruction only writes results for lanes whose EXEC bit is set; disabled
lanes are skipped. Divergent control flow is lowered to EXEC manipulation:

```cpp
// HIP source
if (x[tid] > 0.0f)
    y[tid] = sqrtf(x[tid]);
```

```asm
; conceptual CDNA lowering (gfx942)
v_cmp_lt_f32_e32  vcc, 0, v_x        ; per-lane predicate -> VCC (64-bit)
s_and_saveexec_b64 s[2:3], vcc       ; save old EXEC, EXEC &= VCC (take 'then' lanes)
; ... v_sqrt_f32 etc. run only on enabled lanes ...
s_or_b64          exec, exec, s[2:3] ; restore EXEC (reconverge)
```

Two related per-lane masks matter for kernel writers:

- **VCC** (Vector Condition Code, 64-bit): destination of `v_cmp*` and carry-out
  of vector add/sub. Drives `s_cbranch_vccz/vccnz`.
- **EXEC** (64-bit): the live execution mask. `__ballot(pred)` returns a 64-bit
  value built from each lane's predicate AND its EXEC bit; `__activemask()`
  returns the current EXEC. Both are 64-bit on CDNA — store them in `uint64_t`.

When every lane of a wave takes the same branch the divergent path is skipped
entirely, so structuring work so that all 64 lanes agree is the cheapest way to
avoid serialization.

## Register files: SGPR, VGPR, AGPR

CDNA exposes a scalar register file plus regular and accumulator views of its
vector-register storage:

| File / view | Scope | Per-wave range or namespace | Allocation granularity | Holds |
|---|---|---|---|---|
| **SGPR** | one copy per wave (scalar) | general names `s0`–`s101`; metadata count also includes target specials | GFX9 groups of 16, up to 112 allocated | addresses, loop counts, EXEC/VCC, descriptors |
| **VGPR (ArchVGPR view)** | one 32-bit value per lane | up to 256 names/wave | gfx942 aligns this portion to 4 before combined allocation | per-thread data, addresses, FMA inputs |
| **AGPR (AccVGPR view)** | one 32-bit value per lane | up to 256 names/wave | charged with the regular portion into the combined 8-register allocation | MFMA accumulators |

A few consequences that drive kernel design:

- **Scalars are free-ish.** A value that is uniform across the wave (a base
  pointer, a loop bound) belongs in an SGPR. `v_readfirstlane_b32` and
  `__builtin_amdgcn_readfirstlane` move a uniform VGPR value into the scalar file
  to relieve VGPR pressure.
- **VGPR and AGPR are views of one physical capacity on CDNA2+.** Each SIMD has
  a combined 512-entry-per-lane budget for its resident waves. Regular and
  accumulator namespaces each expose up to 256 names, but occupancy is based on
  their target-specific **combined allocation**, not on two independent
  256-entry banks.
  Accumulators are conventionally kept in the AGPR view so regular names remain
  available for addressing — see [MFMA](mfma.md) and
  [VGPR budgeting](../techniques/vgpr-budgeting.md).
- **Allocation is quantized, and count namespaces must not be mixed.** HSA
  metadata `.vgpr_count` already reports the combined total vector-register
  allocation on both gfx942 and gfx950; `.agpr_count` is the accumulator subset,
  not an extra allocation to add again. Compute
  `round_up(metadata_vgpr_count, 8)`. A lower-level compiler resource remark may
  instead expose separate regular `NumVgprs` and accumulator `NumAgprs`; for
  those separate counts gfx942 and gfx950 derive the total as
  `round_up(round_up(NumVgprs, 4) + NumAgprs, 8)`. Prefer `TotalNumVgprs` when
  the compiler reports it.
- **SGPR claims need their own namespace and unit.** General SGPR names are
  `s0`–`s101` (102 total). The raw per-wave `.sgpr_count` also includes enabled
  target-special pairs such as VCC, FLAT_SCRATCH, and XNACK; it is not rounded.
  GFX9 then rounds allocation to 16-register blocks, which can allocate 112.
  AMDHSA “User SGPR” separately means at most 16 dispatch-initialized registers.
  The community guide's 104-SGPR per-workgroup statement conflates these terms.

## Occupancy: how many waves fit

The vector file and LDS are statically partitioned among resident work. Do not
compare a waves/SIMD quantity directly with a waves/CU quantity. A compact
CDNA3/CDNA4 estimate is:

```text
# HSA metadata .vgpr_count is already the combined vector-register count.
vector_alloc = round_up(metadata_vgpr_count, 8)
sgpr_alloc = round_up(sgpr_count_including_target_specials, 16)

vector_waves_per_simd = min(8, floor(512 / vector_alloc))
vector_waves_per_cu = 4 * vector_waves_per_simd
sgpr_waves_per_simd = min(8, floor(SGPR_capacity_per_SIMD / sgpr_alloc))
sgpr_waves_per_cu = 4 * sgpr_waves_per_simd
lds_waves_per_cu = floor(LDS_per_CU / allocated_LDS_per_workgroup) * waves_per_wg

occupancy_per_cu = min(32, vector_waves_per_cu, sgpr_waves_per_cu,
                       lds_waves_per_cu,
                       other_workgroup_and_scheduler_limits)
```

At a combined allocation of 256 vector registers per lane, a SIMD can hold
**two** waves; an allocation from 264 through 512 permits one wave and does not
by itself imply a spill. Reducing the combined allocation across a residency
threshold can expose another wave. This is the central trade-off behind
[occupancy tuning](../techniques/occupancy-tuning.md) and the
[low-occupancy pattern](../patterns/low-occupancy.md). High occupancy is *not*
always faster — large-tile GEMMs deliberately run at low occupancy and rely on
[software pipelining](../techniques/mfma-pipelining.md) instead.

Inspect the actual allocation in the compiler output or `rocprofv2`:

```bash
# Per-kernel VGPR/SGPR/AGPR and spill counts from the ISA dump
hipcc --save-temps -c gemm.hip -o /dev/null
llvm-objdump -d gemm-hip-amdgcn-amd-amdhsa-gfx942.o | grep -E 'NumVgpr|NumSgpr|NumAgpr|ScratchSize'

# Or annotate occupancy directly
hipcc -Rpass-analysis=kernel-resource-usage -c gemm.hip
```

`scratch`/spill traffic (private memory) appears when allocation cannot satisfy
the live values under the architectural or requested compiler limit. Merely
exceeding 256 combined vector registers is legal and normally lowers the
vector-register limit to one wave/SIMD. Likewise, remapping a regular value to
an unused AGPR index stays in the same physical file and is not a scratch spill.
Confirm actual scratch with the private-segment size, spill counts, and emitted
`scratch_*` instructions.

## Querying width portably (do not hard-code 64)

```cpp
__global__ void reduce_kernel(const float* in, float* out, int n) {
    // warpSize is 64 on gfx9xx (CDNA), 32 or 64 on RDNA — read it, don't assume.
    unsigned long long active = __ballot(threadIdx.x < n);  // 64-bit on CDNA
    int lane = threadIdx.x & (warpSize - 1);
    float v = (threadIdx.x < n) ? in[threadIdx.x] : 0.0f;

    // tree reduction across the wave using shuffles (width = warpSize)
    for (int off = warpSize / 2; off > 0; off >>= 1)
        v += __shfl_down(v, off);

    if (lane == 0) atomicAdd(out, v);
    (void)active;
}
```

Host code can confirm the width with `hipGetDeviceProperties` (`warpSize`
field). Writing `int lane = tid % 32;` is a classic CUDA-port bug on CDNA — it
silently splits a 64-lane wave into two and corrupts cross-lane reductions. See
the [cross-lane page](cross-lane.md) for the reduction primitives themselves.

## Cross-vendor note

A CDNA wave64 is conceptually an NVIDIA warp, but twice as wide (64 vs 32 lanes)
and with **no independent thread scheduling** — reconvergence is explicit EXEC
manipulation, not hardware per-lane PCs. Masks (`EXEC`, `VCC`, `__ballot`) are
64-bit, so a `uint32_t` mask from ported CUDA code is wrong. There is no directly
equivalent architectural AGPR namespace on NVIDIA. CDNA MFMA accumulators
conventionally use the AGPR view of the unified vector file, so they add to the
same occupancy budget as regular per-lane state.

## See also

- [MFMA — Matrix Core instructions](mfma.md)
- [Cross-lane operations (DPP, permute, swizzle)](cross-lane.md)
- [Occupancy tuning](../techniques/occupancy-tuning.md)
- [VGPR budgeting](../techniques/vgpr-budgeting.md)
- [HIP kernel basics](../languages/hip.md)

## Sources

- [AMD CDNA3 "MI300" ISA Reference Guide](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf)
- [HIP Programming — Hardware Capabilities & `warpSize`](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/hardware_features.html)
- [AMD CDNA3 Architecture Whitepaper](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-3-white-paper.pdf)
- [LLVM AMDGPU Backend — User Guide (register allocation, EXEC)](https://llvm.org/docs/AMDGPUUsage.html)
- [Optimizing GEMM on AMD GPUs (occupancy vs. tiling)](https://rocm.blogs.amd.com/artificial-intelligence/matrix-cores/README.html)
