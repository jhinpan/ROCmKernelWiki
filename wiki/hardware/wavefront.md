---
id: hw-wavefront
title: Wavefronts, EXEC Mask & Register Files (CDNA)
type: hardware
architectures:
- gfx942
- gfx950
- gfx1201
tags:
- wave64
- wave32
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
aliases:
- wavefront
- wave64
- warp
- EXEC mask
- occupancy
implemented_by:
- pr-Tensile-1383
---
# Wavefronts, EXEC Mask & Register Files (CDNA)

## Overview

A **wavefront** (AMD's "warp") is the unit of SIMD execution on a CDNA Compute
Unit (CU). On CDNA (gfx9xx, including gfx942/CDNA3 and gfx950/CDNA4) a wavefront
is **always 64 work-items wide — wave64**. There is no wave32 mode on CDNA. RDNA
parts such as gfx1201/RDNA4 support both wave32 and wave64; the kernel-visible
width is reported by `warpSize` and should be queried, never hard-coded.

All 64 lanes of a wavefront share one program counter and step through the same
instruction stream in lockstep. Per-lane divergence is handled by masking, not by
independent PCs (there is no per-lane PC / independent thread scheduling like
NVIDIA Volta+). The active-lane set is tracked by the **EXEC mask**.

A CU is partitioned into **4 SIMD units (SIMD16)**. Each SIMD owns a slice of the
register file and a pool of wave slots; the four pools together hold up to
**40 wavefronts per CU (4 × 10)**. The number actually resident — the
**occupancy** — is almost always limited by VGPR and LDS usage rather than by the
40-wave hardware ceiling.

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

CDNA exposes three register files, all carved out of per-CU SRAM:

| File | Scope | Per-wave limit (gfx942) | Allocation granularity | Holds |
|---|---|---|---|---|
| **SGPR** | one copy per wave (scalar) | ~12.5 KiB/CU total | groups of 16 | addresses, loop counts, EXEC/VCC, descriptors |
| **VGPR (ArchVGPR)** | one 32-bit value per lane | up to 256/wave | groups of 8 dwords | per-thread data, addresses, FMA inputs |
| **AGPR (AccVGPR)** | one 32-bit value per lane | up to 256/wave | groups of 8 dwords | MFMA accumulators |

A few consequences that drive kernel design:

- **Scalars are free-ish.** A value that is uniform across the wave (a base
  pointer, a loop bound) belongs in an SGPR. `v_readfirstlane_b32` and
  `__builtin_amdgcn_readfirstlane` move a uniform VGPR value into the scalar file
  to relieve VGPR pressure.
- **VGPR + AGPR together cap at 512/wave** on CDNA3/CDNA4 (256 Arch + 256 Acc).
  On CDNA3/CDNA4 the MFMA A/B/C/D operands may use either bank, but accumulators
  are conventionally kept in AGPRs so ArchVGPRs are free for addressing — see
  [MFMA](mfma.md) and [VGPR budgeting](../techniques/vgpr-budgeting.md).
- **Allocation is quantized.** VGPRs are allocated in blocks (groups of 8
  dwords), so a kernel using 65 VGPRs is rounded up and may cost an occupancy
  step. Trimming to the next-lower block boundary can add a whole resident wave.

## Occupancy: how many waves fit

The total per-CU SRAM (the VGPR/AGPR file plus 64 kB LDS on gfx942, 160 kB on
gfx950) is statically partitioned among resident waves. Occupancy is the minimum
of several caps:

```text
waves_per_simd = min(
    10,                                    # hardware wave-slot ceiling per SIMD
    floor(VGPRs_per_SIMD / vgprs_per_wave),# vector-register limited
    floor(LDS_per_CU   / lds_per_workgroup) * waves_per_wg / SIMDs  # LDS limited
)
occupancy_per_CU = 4 * waves_per_simd      # 4 SIMDs, max 40 waves/CU
```

The practical lever is `vgprs_per_wave`: at 256 VGPRs/wave a SIMD holds only 1
wave; halving VGPR usage roughly doubles resident waves, restoring the latency
hiding that fills MFMA and memory stalls. This is the central trade-off behind
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

`scratch`/spill traffic (private memory) appears when a wave exceeds 256 VGPRs;
spills serialize on VMEM and usually erase any occupancy benefit, so eliminating
them takes priority over chasing one more resident wave.

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
64-bit, so a `uint32_t` mask from ported CUDA code is wrong. There is no
register-bank equivalent of AGPRs on NVIDIA; tensor-core accumulators on CDNA
live in their own file, which is why MFMA-heavy kernels are AGPR-bound rather
than VGPR-bound.

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
