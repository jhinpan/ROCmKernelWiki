---
id: technique-vgpr-budgeting
title: "VGPR Budgeting — ArchVGPR + AGPR Pressure vs Occupancy"
type: technique
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
---

# VGPR Budgeting — ArchVGPR + AGPR Pressure vs Occupancy

## Why a budget exists

A CDNA wavefront is 64 work-items wide, and every VGPR is therefore 64 lanes ×
4 bytes = 256 bytes of physical register file. A wave can address at most
**512 VGPRs**, split across two banks: **256 ArchVGPRs** (general vector
registers) + **256 AGPRs** (accumulation registers used by the matrix core).
See [the wavefront page](../hardware/wavefront.md) for the register-file layout.

Each CU has four SIMD16 schedulers, and each can hold **up to 10 resident
waves** (40 waves/CU). Residency is almost always capped by **VGPR allocation
and LDS**, not by the wave-slot count. Registers are allocated per wave in
**groups of 8 dwords**, and the ArchVGPR and AGPR banks are budgeted
*independently* — a kernel that is light on ArchVGPRs but heavy on AGPRs is
still limited by whichever bank is fuller.

The practical consequence: **occupancy is a step function of your largest
register bank.** Saving registers only helps if it pushes you across an
allocation threshold.

## Occupancy as a function of register use

For a per-SIMD ArchVGPR file holding `F` registers, the resident wave count is

```
waves_per_simd = min(10, floor(F / round_up(vgprs_per_wave, 8)))
```

and the *same* relation applies independently to the AGPR bank; the kernel runs
at `min(arch_waves, agpr_waves, lds_waves, 10)`. On MI300-class gfx942 the
ArchVGPR file admits up to 10 waves only at very low allocations; large GEMM
tiles routinely sit at 2–4 waves. Illustrative thresholds (read the exact
numbers from your build — see below):

| Larger register bank (per wave) | Waves / SIMD | Notes |
|---|---|---|
| ≤ 64  | 8     | latency-hiding sweet spot for memory-bound kernels |
| 96    | 5     | typical fused-elementwise / norm kernels |
| 128   | 4     | mid-size MFMA accumulator tiles |
| 168   | 3     | large `32×32` accumulator tiles |
| 256   | 2     | maximum tile, minimum latency hiding |

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
tile = **256 AGPRs**, i.e. the entire AGPR bank, leaving exactly two waves and
zero room to grow A/B operand registers. This is the central tension:

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
    // Keep this array small to stay in budget; growing it directly lowers
    // occupancy via the AGPR bank.
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
# Cap ArchVGPRs to force higher occupancy (may induce spills if too low):
hipcc -O3 --offload-arch=gfx942 -mllvm -amdgpu-target-max-vgpr=128 gemm.hip -c

# Inspect the actual allocation the backend chose (the ground truth):
hipcc -O3 --offload-arch=gfx942 --save-temps -c gemm.hip
llvm-objdump -d gemm-hip-amdgcn-amd-amdhsa-gfx942.o | grep -E '\.vgpr_count|\.agpr_count'
```

The Triton AMD backend exposes the same tradeoff through `waves_per_eu` and the
MFMA-size knob `matrix_instr_nonkdim`; raising `waves_per_eu` shrinks the per-
wave register budget exactly as `__launch_bounds__` does.

## Watch for spills

If the requested occupancy demands fewer registers than the kernel needs, the
backend **spills** to scratch (private memory backed by HBM/L2). Scratch traffic
on the hot path is far worse than the occupancy you bought — confirm
`.scratch_size == 0` in the compiled object, or check `SCRATCH` / register-spill
counters in `rocprofv2`/Omniperf. When you see spills together with low
occupancy, you are over-tiled: shrink the accumulator tile or split the K-loop
(see [split-K](split-k.md)) rather than forcing the VGPR cap lower. This is the
exact failure mode catalogued under
[VGPR-pressure](../patterns/vgpr-pressure.md).

## Checklist

1. Compute `acc_regs = (Mt*Nt)/64` and the A/B fragment cost from the
   [matrix calculator](../../sources/refs/ref-matrix-calculator.md).
2. Add addressing + pipeline ArchVGPRs; verify ArchVGPR **and** AGPR banks each
   leave the occupancy you need (`waves = floor(F / round_up(regs,8))`).
3. Set `__launch_bounds__(block, minWaves)` to lock the target in.
4. Build with `--save-temps`; confirm `.vgpr_count`, `.agpr_count`, and
   `.scratch_size == 0`.
5. If memory-bound, prefer ≥4–8 waves; if compute-bound MFMA, 2 waves with a
   deep K-loop is acceptable.

## Sources

- [HIP Programming / Hardware Implementation](https://rocm.docs.amd.com/projects/HIP/en/latest/understand/hardware_implementation.html)
- [AMD CDNA3 ISA Reference Guide](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf)
- [AMD Instinct MI300X Datasheet](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/data-sheets/amd-instinct-mi300x-data-sheet.pdf)
- [Optimizing GEMM on AMD GPUs](https://rocm.blogs.amd.com/artificial-intelligence/matrix-cores/README.html)
- [AMD Matrix Instruction Calculator](https://github.com/ROCm/amd_matrix_instruction_calculator)
