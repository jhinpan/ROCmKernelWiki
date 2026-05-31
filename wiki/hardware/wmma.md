---
id: hw-wmma
title: WMMA — RDNA Wave Matrix Multiply-Accumulate (RDNA4 / gfx1201)
type: hardware
architectures:
- gfx1201
tags:
- wmma
- matrix-core
- wave32
- wave64
- rdna
- fp8
confidence: source-reported
related:
- hw-mfma
- hw-wavefront
- lang-rocwmma
- migration-wmma-vs-mfma
sources:
- doc-llvm-amdgpu
- ref-rocwmma
- blog-amd-matrix-cores
- doc-rocm-hip-hw
aliases:
- WMMA
- wave matrix multiply accumulate
- wave matrix multiply-accumulate
implemented_by:
- pr-aiter-3236
- pr-FlyDSL-250
- pr-composable_kernel-2704
- pr-composable_kernel-2528
- pr-composable_kernel-2466
- pr-FlyDSL-221
- pr-composable_kernel-2319
- pr-aiter-2969
---
# WMMA — RDNA Wave Matrix Multiply-Accumulate (RDNA4 / gfx1201)

## Overview

`v_wmma_*` (Wave Matrix Multiply-Accumulate) is the RDNA family of matrix
instructions, the consumer/workstation counterpart to CDNA's
[MFMA](mfma.md). It computes `D = A · B + C` for a small fixed tile, cooperatively
across an entire wavefront. WMMA debuted on RDNA3 (gfx1100) and is the matrix
path on **RDNA4 (gfx1201)** — the Radeon AI PRO R9700 and RX 9070. RDNA parts do
**not** have MFMA; conversely CDNA parts do not have WMMA. If you are porting a
matrix kernel between an Instinct (CDNA) part and a Radeon (RDNA) part, this is
the instruction-set boundary you cross.

The two biggest structural differences from MFMA come straight from the RDNA
execution model:

1. **Wave width is a choice.** CDNA is wave64-only, but RDNA supports both
   **wave32** and **wave64**. The same logical 16×16 tile is therefore spread
   across either 32 or 64 lanes, which changes how many VGPRs each lane holds for
   a fragment. Query `warpSize` — do not hardcode it.
2. **No AGPRs.** RDNA has a single vector register bank; there are no separate
   accumulation registers (AGPRs). WMMA reads and writes its A/B/C/D fragments
   out of ordinary VGPRs, so matrix tiles compete directly with addressing and
   live state for the one VGPR file.

## Programming WMMA — the portable path

Most kernels should use **[rocWMMA](../languages/rocwmma.md)**, a header-only
C++ fragment API that targets `v_wmma_*` on gfx1100/gfx1201 (and `v_mfma_*` on
CDNA) from one source. The `load → mma_sync → store` flow mirrors CUDA's
`nvcuda::wmma`:

```cpp
#include <rocwmma/rocwmma.hpp>
using namespace rocwmma;

// One 16x16x16 tile: FP16 inputs, FP32 accumulate.
// Launch with a block whose x-dim is a whole wavefront (32 or 64 lanes).
__global__ void wmma_tile_16x16x16(const float16_t* a, const float16_t* b,
                                   float* d, int lda, int ldb, int ldd)
{
    fragment<matrix_a,    16, 16, 16, float16_t, row_major> fragA;
    fragment<matrix_b,    16, 16, 16, float16_t, col_major> fragB;
    fragment<accumulator, 16, 16, 16, float>                fragAcc;

    fill_fragment(fragAcc, 0.0f);
    load_matrix_sync(fragA, a, lda);          // distribute A across the wave
    load_matrix_sync(fragB, b, ldb);          // distribute B across the wave
    mma_sync(fragAcc, fragA, fragB, fragAcc); // emits v_wmma_f32_16x16x16_f16
    store_matrix_sync(d, fragAcc, ldd, mem_row_major);
}
```

rocWMMA hides the per-wave-width register layout: the *same* fragment code
compiles for wave32 or wave64, and the library picks the matching `v_wmma_*`
encoding and lane mapping.

## Programming WMMA — the intrinsic path

When you need full control you can emit the instruction directly via a Clang
builtin / LLVM intrinsic. The builtin name carries the wave width (`_w32` /
`_w64`), because the operand vector lengths differ between the two:

```cpp
// gfx12 (RDNA4), wave32: 16x16x16 FP16 -> FP32.
// In wave32 the 16x16 FP32 accumulator (256 elems / 32 lanes) = 8 VGPRs/lane.
using half16 = __attribute__((ext_vector_type(16))) __fp16;
using float8 = __attribute__((ext_vector_type(8)))  float;

__device__ float8 wmma_f16_w32(half16 a, half16 b, float8 c) {
    return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
}
```

The corresponding LLVM intrinsic is `llvm.amdgcn.wmma.f32.16x16x16.f16`; the
[AMDGPU backend docs](https://llvm.org/docs/AMDGPUUsage.html) list the full set
of gfx11/gfx12 WMMA intrinsics, their operand vector types, and the
wave32/wave64 variants.

> **Fragment register count scales with wave width.** A 16×16 FP32 accumulator
> holds 256 elements. In **wave32** that is 256 / 32 = **8 VGPRs per lane**; in
> **wave64** it is 256 / 64 = **4 VGPRs per lane**. Wider waves cut the
> per-lane accumulator footprint but halve the lanes available to feed the
> matrix unit. Choosing the wave width is therefore an occupancy/throughput
> trade — see [wavefront & occupancy](wavefront.md).

## Shapes and dtypes

WMMA uses a fixed **K = 16** base shape (`16×16×16`) for 16-bit and smaller
inputs, in contrast to MFMA's menu of 16×16 and 32×32 tiles with type-dependent
K. The RDNA3 baseline set (also present on gfx1201):

| Input dtype | Acc / out | Shape (M×N×K) | Example mnemonic |
|---|---|---|---|
| FP16 | FP32 | 16×16×16 | `v_wmma_f32_16x16x16_f16` |
| BF16 | FP32 | 16×16×16 | `v_wmma_f32_16x16x16_bf16` |
| FP16 | FP16 | 16×16×16 | `v_wmma_f16_16x16x16_f16` |
| BF16 | BF16 | 16×16×16 | `v_wmma_bf16_16x16x16_bf16` |
| INT8 (iu8) | INT32 | 16×16×16 | `v_wmma_i32_16x16x16_iu8` |
| INT4 (iu4) | INT32 | 16×16×16 | `v_wmma_i32_16x16x16_iu4` |

RDNA4 (gfx1201) extends this with low-precision floating-point support —
notably **OCP FP8** (`E4M3`/`E5M2`) inputs accumulating to FP32, and wider-K
8-bit/4-bit shapes — bringing the Radeon matrix path closer to the CDNA4 FP8
story. Because the exact gfx1201 shape/encoding table is best read from the
generated backend, confirm a given mnemonic against the AMDGPU intrinsic list
rather than assuming it from the RDNA3 set above; that part is marked
`source-reported` here.

> **RDNA3 input replication vs RDNA4.** On RDNA3, WMMA fed identical A/B operand
> data into the upper and lower halves of the wave (lanes 0–15 mirrored 16–31 in
> wave32), so the fragment "load" duplicated inputs. RDNA4 reworked the operand
> layout. rocWMMA abstracts this, but hand-written intrinsic kernels ported from
> gfx1100 to gfx1201 must re-derive their lane→element mapping — do not assume
> the RDNA3 packing carries over.

## WMMA vs MFMA at a glance

| | WMMA (RDNA4 / gfx1201) | MFMA (CDNA3/4 / gfx942/gfx950) |
|---|---|---|
| Wave width | wave32 **and** wave64 | wave64 only |
| Accumulator regs | ordinary VGPRs | AGPRs (separate bank) |
| Base shapes | 16×16×16 (fixed K=16 for ≤16-bit) | 16×16 and 32×32, type-dependent K |
| Encoding | VOP3P (`v_wmma_*`) | VOP3P-MAI (`v_mfma_*`) |
| Sparsity | structured-sparse WMMA variants | `v_smfmac_*` (4:2) |
| Target HW | Radeon AI PRO / RX 9000 | Instinct MI300/MI350 |

For a deeper porting walkthrough see
[WMMA vs MFMA migration](../migration/wmma-vs-mfma.md).

## Practical notes

- **Pick the wave width deliberately.** Compile with `-mwavefrontsize64` (or the
  HIP launch/codegen equivalent) only after measuring; wave32 often wins on RDNA
  for matrix-light, latency-bound kernels, while wave64 reduces per-lane
  accumulator VGPRs for big tiles.
- **VGPR budget is the ceiling.** With no AGPR bank, large WMMA accumulator
  tiles eat the same register file as your loads and addressing. Treat
  accumulator tile size as an occupancy knob — see
  [VGPR pressure](../patterns/vgpr-pressure.md).
- **Feed WMMA from LDS.** As with MFMA, stage A/B tiles through
  [LDS](lds.md) with [double buffering](../techniques/lds-double-buffering.md) so
  the matrix unit is not stalled on global memory.

## See also

- [MFMA — CDNA matrix cores](mfma.md)
- [rocWMMA language guide](../languages/rocwmma.md)
- [WMMA vs MFMA migration](../migration/wmma-vs-mfma.md)
- [Wavefront, EXEC & occupancy](wavefront.md)

## Sources

- [LLVM AMDGPU Backend User Guide — WMMA intrinsics](https://llvm.org/docs/AMDGPUUsage.html)
- [rocWMMA — header-only C++ matrix fragment API](https://github.com/ROCm/rocWMMA)
- [AMD Matrix Cores (programming overview)](https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores/README.html)
- [ROCm HIP hardware capabilities / GPU ISA reference](https://rocm.docs.amd.com/projects/HIP/en/latest/)
