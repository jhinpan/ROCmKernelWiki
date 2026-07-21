---
id: technique-vgpr-budgeting
title: VGPR Budgeting — ArchVGPR + AGPR Pressure vs Occupancy
type: technique
version_sensitive:
- vs-wave-slots-cdna3-cdna4
- vs-cdna-unified-vgpr-agpr-allocation
architectures:
- gfx942
- gfx950
tags:
- vgpr-budgeting
- agpr-management
- occupancy-tuning
- vgpr
- agpr
- mfma
- register-blocking
confidence: source-reported
reproducibility: snippet
hardware_features:
- vgpr
- agpr
- wave64
- matrix-core
kernel_types:
- gemm
- flash-attention
languages:
- hip
- gcn-asm
related:
- hw-wavefront
- hw-mfma
- technique-occupancy-tuning
- technique-mfma-pipelining
- pattern-vgpr-pressure
sources:
- doc-rocm-hip-hw
- doc-cdna3-isa
- doc-mi300x-datasheet
- blog-gemm-optimization
- ref-matrix-calculator
- blog-amdgpu-kernel-opt-guide
implemented_by:
- pr-Tensile-1383
- pr-sglang-25898
- pr-composable_kernel-2528
- pr-composable_kernel-2466
- pr-composable_kernel-2276
- pr-composable_kernel-2110
- pr-FlyDSL-447
- pr-sglang-26208
---
# VGPR Budgeting — ArchVGPR + AGPR Pressure vs Occupancy

## Why a budget exists

A CDNA wavefront is 64 work-items wide, and every VGPR is therefore 64 lanes ×
4 bytes = 256 bytes of physical register file. On CDNA2 and later, each SIMD
has one **combined 512-entry-per-lane vector-register capacity** shared by the
regular **ArchVGPR** view and the accumulator **AccVGPR/AGPR** view. A wave can
name up to 256 registers in each view, but the two allocated counts are added
for residency; they are not independent 256-entry occupancy pools. See
[the wavefront page](../hardware/wavefront.md) for the register-file layout.

Each CU has four SIMD16 schedulers, and each can hold **up to 8 resident
waves** (32 waves/CU) on CDNA3/CDNA4. Residency is almost always capped by
**VGPR allocation and LDS**, not by the wave-slot count. Regular and
accumulator use is combined into one per-wave allocation, encoded in **groups
of 8 dwords**, and charged against the 512-entry capacity.

The practical consequence: **occupancy is a step function of the combined
regular-plus-accumulator allocation.** Saving registers only helps if it pushes
that sum across a residency threshold.

## Occupancy as a function of register use

For CDNA3/CDNA4, use counts expressed as 32-bit registers per lane. The metadata
accounting differs slightly by target:

```
# gfx942: align the regular portion to 4, add the accumulator count,
# then encode the combined allocation in groups of 8.
vector_alloc_gfx942 = round_up(
    round_up(arch_vgprs_per_lane, 4) + accum_vgprs_per_lane, 8)

# gfx950: .vgpr_count is already the combined raw count.
vector_alloc_gfx950 = round_up(vgpr_count, 8)

waves_per_simd = min(8, floor(512 / vector_alloc_for_target))
```

On gfx942, use the separate `.vgpr_count` and `.agpr_count` in the first formula;
rounding both independently to eight would overestimate allocation. On gfx950,
`.vgpr_count` already includes both, so do **not** add `.agpr_count` a second
time. Large GEMM tiles routinely sit at 2–4 waves/SIMD. Illustrative combined
thresholds (read the exact numbers from your build — see below):

| Combined vector allocation (per lane) | Waves / SIMD | Notes |
|---|---|---|
| ≤ 64  | 8     | latency-hiding sweet spot for memory-bound kernels |
| 96    | 5     | typical fused-elementwise / norm kernels |
| 128   | 4     | mid-size MFMA accumulator tiles |
| 168   | 3     | large `32×32` accumulator tiles |
| 256   | 2     | two waves still fit exactly |
| 264–512 | 1   | legal allocation; not by itself evidence of a spill |

Two resident waves per SIMD is often *fine* for a compute-bound MFMA GEMM —
the matrix pipe is kept busy by deep K-loop unrolling, not by wave swapping —
but it leaves almost no slack to hide a stall, so a single spill or an
unscheduled `s_waitcnt` becomes very expensive.

## The accumulator-tiling tradeoff

The dominant ArchVGPR/AGPR consumer in a GEMM is the **C/D accumulator tile**,
which lives in AGPRs (see [MFMA](../hardware/mfma.md)). The accumulator for an
`Mt × Nt` per-wave output tile, in FP32, costs

```
acc_regs = (Mt * Nt) / 64        # 64 lanes hold one element each
```

So a `64×64` per-wave tile = 4096 elements / 64 = **64 AGPRs**; a `128×128`
tile = **256 AGPRs**. The accumulator alone leaves room for two waves, but the
regular registers needed for A/B operands and addressing make the combined
allocation exceed 256 and normally reduce the vector-register limit to one
wave/SIMD. This is the central tension:

- **Bigger accumulator tile** → more MACs per byte of A/B loaded (better
  arithmetic intensity, fewer LDS round-trips) **but** more AGPRs → lower
  occupancy → less latency hiding.
- **Smaller accumulator tile** → higher occupancy, but more redundant operand
  traffic and more loop overhead.

The sweet spot is the largest tile that *still* leaves enough ArchVGPRs for the
A/B fragments, addressing, and the software pipeline, while keeping at least the
occupancy your memory latency requires. Use the
[Matrix Instruction Calculator](../../sources/refs/ref-matrix-calculator.md) to
get exact A/B/C/D GPR counts per MFMA shape before committing to a tile.

## Controlling the budget from HIP

Two levers cap or shape register allocation:

```cpp
#include <hip/hip_runtime.h>

// __launch_bounds__(maxThreadsPerBlock, minWavesPerEU)
// The second argument asks the backend to allocate few enough VGPRs that at
// least N waves/EU(SIMD) stay resident -- a hard hint that trades tile size
// for occupancy.
template <int BLOCK = 256>
__global__ void __launch_bounds__(BLOCK, 4)   // target >= 4 waves/SIMD
gemm_tile(const __fp16* __restrict__ A,
          const __fp16* __restrict__ B,
          float* __restrict__ C, int M, int N, int K)
{
    using float4 = __attribute__((__vector_size__(4 * sizeof(float)))) float;
    using half4  = __attribute__((__vector_size__(4 * sizeof(__fp16)))) __fp16;

    // Per-wave accumulator: 4x 16x16 MFMA tiles -> 4 * 4 = 16 AGPRs (FP32).
    // Keep this array small to stay in the combined vector-register budget;
    // growing it directly lowers occupancy.
    float4 acc[4] = {};

    half4 a_frag, b_frag;   // operand fragments live in ArchVGPRs
    // ... staged loads into a_frag / b_frag from LDS ...
    #pragma unroll
    for (int t = 0; t < 4; ++t)
        acc[t] = __builtin_amdgcn_mfma_f32_16x16x16f16(a_frag, b_frag, acc[t], 0, 0, 0);

    // ... epilogue store of acc -> C ...
}
```

Compiler-side controls (Clang / ROCm):

```bash
# Apply an explicit, compiler-specific ArchVGPR cap (may induce spills if too low):
hipcc -O3 --offload-arch=gfx942 -mllvm -amdgpu-target-max-vgpr=128 gemm.hip -c

# Inspect the actual allocation the backend chose (the ground truth):
hipcc -O3 --offload-arch=gfx942 --save-temps -c gemm.hip
llvm-objdump -d gemm-hip-amdgcn-amd-amdhsa-gfx942.o | grep -E '\.vgpr_count|\.agpr_count'
```

The `128` above is an explicit tuning choice. The captured community guide's
claim that 128 is the default cap is compiler/version-specific, not a CDNA3 or
CDNA4 architectural limit.

The Triton AMD backend exposes the same tradeoff through `waves_per_eu` and the
MFMA-size knob `matrix_instr_nonkdim`; raising `waves_per_eu` shrinks the per-
wave register budget exactly as `__launch_bounds__` does.

## Watch for spills

If the requested occupancy demands fewer registers than the kernel needs, the
backend **spills** to scratch (private memory backed by HBM/L2). Scratch traffic
on the hot path is far worse than the occupancy you bought — confirm
`ScratchSize == 0` / `.private_segment_fixed_size == 0` in the compiled object,
or check scratch instructions and register-spill counters. A combined allocation
above 256 is still legal and merely limits the SIMD to one resident wave; it is
not proof of spilling. Moving a regular live range into an unused AGPR index is
register remapping inside the same physical file, also not a scratch spill.
When actual scratch spills appear together with low occupancy, shrink the
accumulator tile or split the K-loop (see [split-K](split-k.md)) rather than
forcing the VGPR cap lower. This is the exact failure mode catalogued under
[VGPR-pressure](../patterns/vgpr-pressure.md).

## Checklist

1. Compute `acc_regs = (Mt*Nt)/64` and the A/B fragment cost from the
   [matrix calculator](../../sources/refs/ref-matrix-calculator.md).
2. Compute the target-specific combined allocation above; verify it leaves the
   occupancy you need (`waves = floor(512 / vector_alloc)`).
3. Set `__launch_bounds__(block, minWaves)` to lock the target in.
4. Build with `--save-temps`; interpret `.vgpr_count`/`.agpr_count` for the
   target and confirm the scratch/private-segment size is zero.
5. If memory-bound, prefer ≥4–8 waves/SIMD; if compute-bound MFMA, 2 waves/SIMD
   with a deep K-loop can be acceptable.

## Sources

- [HIP Programming / Hardware Implementation](https://rocm.docs.amd.com/projects/HIP/en/latest/understand/hardware_implementation.html)
- [AMD CDNA3 ISA Reference Guide](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf)
- [AMD Instinct MI300X Datasheet](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/data-sheets/amd-instinct-mi300x-data-sheet.pdf)
- [Optimizing GEMM on AMD GPUs](https://rocm.blogs.amd.com/artificial-intelligence/matrix-cores/README.html)
- [AMD Matrix Instruction Calculator](https://github.com/ROCm/amd_matrix_instruction_calculator)
- [AMDGPU Kernel Optimization Guide (captured snapshot)](https://github.com/nod-ai/amd-shark-ai/blob/efa471aeef66a260c85983cc41e833bfa769dade/docs/amdgpu_kernel_optimization_guide.md)
