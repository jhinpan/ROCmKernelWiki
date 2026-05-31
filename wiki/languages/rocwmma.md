---
id: lang-rocwmma
title: rocWMMA — Portable C++ Matrix Fragment API
type: language
architectures:
- gfx90a
- gfx942
- gfx950
- gfx1100
- gfx1201
tags:
- cpp
- hip
- mfma
- wmma
- matrix-core
- bf16
- fp16
- fp8
confidence: source-reported
reproducibility: snippet
languages:
- cpp
- hip
related:
- hw-mfma
- hw-wmma
- migration-wmma-vs-mfma
- lang-composable-kernel
- kernel-ck-hgemm
sources:
- ref-rocwmma
- hw-mfma
- hw-wmma
- blog-amd-matrix-cores
- doc-cdna3-isa
implemented_by:
- pr-composable_kernel-2110
- pr-triton-775
- pr-composable_kernel-932
- pr-composable_kernel-690
- pr-composable_kernel-2963
- pr-composable_kernel-2821
- pr-composable_kernel-2606
- pr-composable_kernel-2528
---
# rocWMMA — Portable C++ Matrix Fragment API

## Overview

**rocWMMA** is AMD's header-only C++ library for programming matrix cores through
a *fragment* abstraction, deliberately modeled on NVIDIA's `nvcuda::wmma` API. A
kernel declares opaque `fragment` objects, fills them with `load_matrix_sync`,
multiplies with `mma_sync`, and writes results with `store_matrix_sync`. The
library lowers those calls to the right hardware instruction for the target GPU:

- `v_mfma_*` **MFMA** instructions on CDNA (gfx90a / gfx942 / gfx950) — see
  [the MFMA page](../hardware/mfma.md).
- `v_wmma_*` **WMMA** instructions on RDNA (gfx1100 / gfx1201) — see
  [the WMMA page](../hardware/wmma.md).

The payoff is **source portability**: one fragment-based GEMM epilogue compiles
to MI300X, MI350X, and Radeon RX 9070 with no instruction-specific code. The cost
is that the fragment layout is opaque, so you cannot hand-tune the per-lane
register mapping the way you can when calling `__builtin_amdgcn_mfma_*` directly.
rocWMMA is the right tool when you want readable, portable matrix-core code; reach
for [Composable Kernel](composable-kernel.md) or raw builtins when you need full
control of the LDS/VGPR layout for a peak-performance GEMM.

It is header-only (`#include <rocwmma/rocwmma.hpp>`) and MIT-licensed; everything
runs inside ordinary HIP `__global__` kernels.

## Programming model

The five core entry points mirror the CUDA WMMA names exactly:

| Call | Purpose |
|---|---|
| `fragment<Use, M, N, K, T, Layout>` | Typed tile handle (register-resident) |
| `fill_fragment(frag, v)` | Broadcast-initialize (e.g. zero the accumulator) |
| `load_matrix_sync(frag, ptr, ld)` | Cooperative wavefront load from global/LDS |
| `mma_sync(d, a, b, c)` | `D = A·B + C` on the matrix core |
| `store_matrix_sync(ptr, frag, ld, layout)` | Cooperative wavefront store |

`Use` is one of `matrix_a`, `matrix_b`, or `accumulator`. `Layout` is
`row_major` or `col_major`. As with MFMA itself, every fragment is distributed
across the **whole wavefront** (64 lanes on CDNA, 32 or 64 on RDNA), so these are
collective calls — all lanes must participate.

## Single-tile GEMM example

A minimal block-tile multiply that each wavefront performs on a 16×16×16 tile.
This compiles unchanged for gfx942 (→ `v_mfma_f32_16x16x16_f16`) and gfx1201
(→ a `v_wmma_*` sequence):

```cpp
#include <hip/hip_runtime.h>
#include <rocwmma/rocwmma.hpp>

using namespace rocwmma;

constexpr int M = 16, N = 16, K = 16;

// One wavefront computes a 16x16 output tile: C[16x16] = A[16xK] * B[Kx16]
__global__ void wmma_tile_gemm(const __half* A, const __half* B, float* C,
                               int lda, int ldb, int ldc, int Kdim)
{
    // Fragment declarations — opaque, register-resident, per-wavefront
    fragment<matrix_a,    M, N, K, __half, row_major> fragA;
    fragment<matrix_b,    M, N, K, __half, col_major> fragB;
    fragment<accumulator, M, N, K, float>             fragAcc;

    fill_fragment(fragAcc, 0.0f);

    // March along the K dimension, accumulating into fragAcc
    for (int k = 0; k < Kdim; k += K) {
        load_matrix_sync(fragA, A + k,        lda);   // A[:, k:k+K]
        load_matrix_sync(fragB, B + k,        ldb);   // B[k:k+K, :]
        mma_sync(fragAcc, fragA, fragB, fragAcc);     // D = A*B + C
    }

    // Write the 16x16 result tile back to global memory
    store_matrix_sync(C, fragAcc, ldc, mem_row_major);
}
```

Launch one wavefront per output tile (`warpSize` is 64 on CDNA, so a 64-thread
block is exactly one wave):

```cpp
dim3 block(warpSize);                 // one wavefront
dim3 grid((Ncols + N - 1) / N,
          (Mrows + M - 1) / M);
wmma_tile_gemm<<<grid, block>>>(dA, dB, dC, lda, ldb, ldc, Kdim);
```

Compile for one or several targets at once:

```bash
hipcc -O3 --offload-arch=gfx942 --offload-arch=gfx1201 \
      -I/opt/rocm/include wmma_gemm.cpp -o wmma_gemm
```

## Supported shapes and types

rocWMMA exposes the matrix-core shapes that the underlying ISA provides. On CDNA3
the FP16/BF16 path is the familiar `16×16×16` and `32×32×8`, accumulating into
FP32, exactly matching the MFMA dense shapes (FP8 widens K, e.g. `16×16×32`).
RDNA4 WMMA uses its own tile geometry. Because the geometry differs across
families, **portable code should parameterize over `M/N/K`** and avoid assuming a
specific block shape.

| Family | Instruction | Typical FP16→FP32 tile |
|---|---|---|
| CDNA3 gfx942 | `v_mfma_f32_16x16x16_f16` | 16×16×16, 32×32×8 |
| CDNA4 gfx950 | `v_mfma_*` (incl. wider-K, f8f6f4) | 16×16×16, 16×16×32 |
| RDNA4 gfx1201 | `v_wmma_*` | wave32 / wave64 variants |

Supported element types track the hardware: `__half`, `hip_bfloat16`, FP8/BF8
(FNUZ on gfx942, OCP on gfx950), `int8_t`→`int32_t`, and `float`/`double` on
CDNA. The same source picks up new precisions only where the target silicon has
them — there is no software emulation fallback for a missing matrix shape.

## Performance and tuning notes

- **rocWMMA is a productivity/portability layer, not an auto-tuner.** A naive
  one-wave-per-tile kernel like the one above is correct but leaves throughput on
  the table; it neither stages A/B through [LDS](../hardware/lds.md) nor
  [pipelines the MFMA issue](../techniques/mfma-pipelining.md). Expect to add
  cooperative LDS loads, multi-tile register blocking, and double buffering to
  approach the [MI300X peak figures](../hardware/mfma.md).
- The fragment layout is opaque, so **cross-lane shuffles on raw fragment data
  are undefined** — operate on data before loading or after storing, not on the
  in-flight fragment.
- `mma_sync` consumes accumulators in AGPRs on CDNA; large accumulator tiles are
  AGPR-bound and trade against occupancy (see
  [VGPR/AGPR pressure](../patterns/vgpr-pressure.md)).
- For top-end GEMM/attention, libraries built on raw builtins (Composable Kernel,
  Tensile/hipBLASLt) will outperform a hand-written rocWMMA loop. rocWMMA shines
  for custom fused epilogues and CUDA-WMMA ports.

## Porting from CUDA `nvcuda::wmma`

Because the API names and call sequence match, most CUDA WMMA kernels port by
swapping the namespace and header:

```cpp
// CUDA:                                 // rocWMMA:
#include <mma.h>                         #include <rocwmma/rocwmma.hpp>
using namespace nvcuda::wmma;            using namespace rocwmma;
```

The semantic gotchas are wavefront width (64 on CDNA vs 32 on NVIDIA — query
`warpSize`, never hardcode 32) and the per-family tile shapes. See
[WMMA vs MFMA](../migration/wmma-vs-mfma.md) for the full mapping.

## See also

- [MFMA — AMD Matrix Core Instructions](../hardware/mfma.md)
- [WMMA — RDNA matrix instructions](../hardware/wmma.md)
- [Composable Kernel / ck_tile](composable-kernel.md)
- [CK FP16 GEMM kernel](../kernels/ck-hgemm.md)

## Sources

- [rocWMMA — ROCm/rocWMMA (GitHub)](https://github.com/ROCm/rocWMMA)
- [rocWMMA documentation (rocm.docs.amd.com)](https://rocm.docs.amd.com/projects/rocWMMA/en/latest/)
- [AMD Matrix Cores (ROCm Blogs)](https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores/README.html)
- [AMD Instinct MI300/CDNA3 ISA Reference Guide](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf)
