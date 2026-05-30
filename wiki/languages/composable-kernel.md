---
id: lang-composable-kernel
title: "Composable Kernel (CK / ck_tile) — A Tile DSL for CDNA"
type: language
architectures:
- gfx942
- gfx950
- gfx1201
tags:
- composable-kernel
- cpp
- hip
- mfma
- gemm
- flash-attention
- register-blocking
- lds-double-buffering
confidence: source-reported
reproducibility: snippet
languages:
- composable-kernel
- hip
- cpp
related:
- hw-mfma
- lang-rocwmma
- lang-triton-amd
- kernel-ck-hgemm
- kernel-flash-attention-ck
- technique-mfma-pipelining
sources:
- ref-composable-kernel
- blog-cktile-gemm
- blog-cktile-flash
- blog-amd-matrix-cores
- doc-rocm-hip-hw
---

# Composable Kernel (CK / ck_tile) — A Tile DSL for CDNA

## Overview

**Composable Kernel (CK)** is AMD's open-source (MIT) HIP C++ template library
for high-performance GPU operators. It is the production backend behind many
ROCm operators — GEMM, fused GEMM epilogues, convolution, FlashAttention,
fused-MoE, normalization — and is consumed in turn by hipBLASLt, MIOpen, and
[AITER](../../sources/refs/ref-aiter.md).

CK comes in two layers:

- **Legacy CK** — a deep template hierarchy (`DeviceGemm*`, `GridwiseGemm*`,
  `BlockwiseGemm*`) parameterized by tile sizes and an `MfmaInstruction`. Very
  fast, but the templates are dense and hard to compose.
- **ck_tile** — a newer, lighter "tile programming" DSL that exposes the
  hardware as composable **tile operators** organized into tiers
  (warp → block → pipeline → kernel), built around **distributed tensors** and
  explicit `load_tile` / `store_tile` / `shuffle_tile` primitives. This page
  focuses on ck_tile.

Everything ultimately lowers to HIP and emits
[`v_mfma_*` matrix-core instructions](../hardware/mfma.md) on CDNA (and `v_wmma_*`
on RDNA4 gfx1201).

## The four tiers

ck_tile structures a kernel as nested operators, each tier responsible for a
mapping level of the GPU hierarchy:

| Tier | Scope | Responsibility |
|---|---|---|
| **warp** | one wavefront (64 lanes) | wraps a single MFMA/WMMA shape; defines the per-wave A/B/C register distribution |
| **block** | one workgroup (LDS) | tiles the warp op over a block tile; manages LDS staging and `__syncthreads` |
| **pipeline** | one block over the K-loop | software pipelining: prefetch (direct-to-LDS), double-buffer, MFMA/global overlap |
| **kernel** | the grid | tile-to-workgroup scheduling (incl. [stream-K](../techniques/stream-k.md)), epilogue/fusion, launch config |

Because each tier is an independent template policy, you can swap a pipeline
(e.g. a 2-stage prefetch vs. a ping-pong schedule) without rewriting the warp- or
block-level math.

## Distributed tensors and tile coordinates

The core abstraction is the **distributed tensor**: a logical tile whose elements
are spread across the threads (and registers) of a tile of the hardware according
to a *tile distribution*. A tile distribution is a composition of **tensor
coordinate transforms** (`merge`, `unmerge`, `embed`, `replicate`) that maps the
logical `(M, N, K)` index space onto the physical `(warp, lane, register)` space.

This is what lets the same algorithm target both a 16×16×16 and a 32×32×8 MFMA:
you change the warp-tier distribution, not the loop body. The element→register
layout for any given MFMA shape can be cross-checked with the
[AMD Matrix Instruction Calculator](../../sources/refs/ref-matrix-calculator.md).

The tile-level data movement primitives are:

- `load_tile` / `async_load_tile` — gather a global/LDS window into a distributed
  tensor; the async form lowers to [direct-to-LDS](../hardware/async-copy-lds.md)
  copies that bypass VGPRs.
- `store_tile` — scatter a distributed tensor back to global or LDS.
- `shuffle_tile` — re-distribute a tensor across lanes/registers (e.g. to convert
  an MFMA C-fragment layout into a coalesced store layout) using cross-lane ops.

## A minimal ck_tile GEMM

The snippet below sketches a block-tier GEMM body in the ck_tile style: declare
the per-block tile shapes, stage A/B through LDS, run the warp MMA over the
K-loop, then store. (Type/policy names follow the upstream
[`ck_tile` GEMM example](https://github.com/ROCm/composable_kernel); tune the
tile constants for your problem.)

```cpp
#include "ck_tile/core.hpp"
#include "ck_tile/ops/gemm.hpp"

using namespace ck_tile;

// Block tile: 128x128 output, K-step 32; one MFMA warp shape underneath.
using GemmShape = TileGemmShape<
    sequence<128, 128, 32>,   // M x N x K per block
    sequence<2, 2, 1>,        // warp arrangement (MWarp x NWarp)
    sequence<32, 32, 16>>;    // warp MMA tile  -> v_mfma_f32_32x32x16_*

template <typename Pipeline, typename Epilogue>
__global__ void ck_gemm_kernel(GemmKernelArgs args)
{
    // 1) Make distributed tensors over global memory (with OOB-safe windows).
    auto a_win = make_tile_window(args.a_view, GemmShape::kM_kK, block_origin_m());
    auto b_win = make_tile_window(args.b_view, GemmShape::kN_kK, block_origin_n());

    // 2) Block/pipeline tier drives the K-loop: prefetch -> LDS -> MFMA.
    //    The pipeline policy chooses double-buffering + async (direct-to-LDS).
    auto c_block = Pipeline{}(a_win, b_win, args.K, smem_ptr());

    // 3) Epilogue: shuffle the accumulator into a coalesced layout, then store.
    auto c_win = make_tile_window(args.c_view, GemmShape::kM_kN, block_origin_mn());
    Epilogue{}(c_win, shuffle_tile(c_block));
}
```

A complete, launchable FP16 GEMM built on these pieces is documented on the
[CK HGEMM kernel page](../kernels/ck-hgemm.md); the FlashAttention-2 forward
pipeline assembled the same way is on the
[CK FlashAttention page](../kernels/flash-attention-ck.md).

## The pipeline tier is where performance lives

The block math (LDS tiling + MFMA) is largely mechanical; the **pipeline policy**
is what separates a roofline-bound kernel from a mediocre one. ck_tile pipelines
typically combine:

- **Double-buffered LDS** so the next K-slice is being filled while the current
  one feeds the matrix core — see
  [LDS double buffering](../techniques/lds-double-buffering.md).
- **`async_load_tile`** to issue HBM→LDS copies that bypass VGPRs and overlap
  streaming with compute, gated by [`s_waitcnt vmcnt`](../hardware/s-waitcnt.md).
- **MFMA pipelining / ping-pong** to keep the matrix unit busy across iterations,
  the same idea covered under [MFMA pipelining](../techniques/mfma-pipelining.md).

On gfx950, `async_load_tile` can use the wider `global_load_lds_dwordx3/x4`
copies; on gfx942 it uses the 32-bit direct-to-LDS path. ck_tile selects this per
target, so the same kernel source recompiles for CDNA3, CDNA4, and RDNA4
(gfx1201, where it emits WMMA instead of MFMA).

## Building

CK is header-heavy and instantiates one device-function object per tile config,
so compiles are slow — instantiate only the shapes you need.

```bash
git clone https://github.com/ROCm/composable_kernel.git
cd composable_kernel && mkdir build && cd build
cmake -D CMAKE_CXX_COMPILER=hipcc \
      -D GPU_TARGETS="gfx942;gfx950" \
      -D CMAKE_BUILD_TYPE=Release ..
# Build a single ck_tile example rather than the whole library:
make tile_example_gemm_basic -j
```

## When to use CK vs. alternatives

- Use **ck_tile** when you need a custom fused operator (e.g. GEMM + bespoke
  epilogue, or a non-standard attention variant) and want full control of the
  tiling/pipeline while staying in portable HIP C++.
- Use [hipBLASLt](../../sources/refs/ref-hipblaslt.md) (CK/Tensile-backed) for
  drop-in GEMM with autotuning, or [Triton](../languages/triton-amd.md) for
  faster iteration with less boilerplate.
- Use [rocWMMA](../languages/rocwmma.md) when you only need a thin
  fragment-level wrapper over a single MMA shape, not a full pipeline.

## Sources

- [ROCm/composable_kernel (GitHub)](https://github.com/ROCm/composable_kernel)
- [CK-tile: a GEMM walkthrough (ROCm Blogs)](https://rocm.blogs.amd.com/)
- [FlashAttention with CK-tile (ROCm Blogs)](https://rocm.blogs.amd.com/)
- [AMD Matrix Cores (ROCm Blogs)](https://rocm.blogs.amd.com/software-tools-optimization/amd-matrix-cores/README.html)
- [HIP / ROCm hardware programming guide](https://rocm.docs.amd.com/)
