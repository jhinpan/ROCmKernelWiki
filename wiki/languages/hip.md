---
id: lang-hip
title: HIP — Kernel Basics, LDS, and AMDGCN Builtins
type: language
version_sensitive:
- vs-permlane16-gfx950
architectures:
- gfx942
- gfx950
tags:
- hip
- cpp
- lds
- wave64
- wave32
- mfma
- permute
- co-issue
confidence: source-reported
reproducibility: snippet
languages:
- hip
- cpp
related:
- hw-mfma
- hw-wavefront
- hw-lds
- hw-cross-lane
- technique-wave-reduce
- migration-cuda-to-hip
sources:
- doc-rocm-hip-hw
- doc-llvm-amdgpu
- blog-amd-matrix-cores
- blog-matrix-cores-cdna
implemented_by:
- pr-aiter-2136
- pr-composable_kernel-2723
- pr-composable_kernel-2722
- pr-composable_kernel-2606
- pr-composable_kernel-2528
- pr-aiter-2394
- pr-Tensile-1521
- pr-Tensile-1383
---
# HIP — Kernel Basics, LDS, and AMDGCN Builtins

## Overview

**HIP** (Heterogeneous-compute Interface for Portability) is AMD's C++ kernel
language. Syntactically it mirrors CUDA — `__global__` entry points, a `<<<>>>`
launch grammar, `__shared__` memory, `threadIdx`/`blockIdx` — so most CUDA
kernels port mechanically. What differs is the *hardware underneath*: kernels are
compiled by Clang/LLVM to the **AMDGCN** ISA, execute as **wavefronts**
(64 lanes on CDNA, 32 or 64 on RDNA), and reach matrix cores and cross-lane
hardware through `__builtin_amdgcn_*` intrinsics rather than CUDA's `mma`/PTX.

This page covers the load-bearing primitives a kernel engineer actually touches:
the function qualifiers, the two launch syntaxes, LDS via `__shared__`,
`__syncthreads`, the `warpSize` portability trap, and the AMDGCN builtins for
MFMA, cross-lane permute, and instruction scheduling.

## A minimal kernel and the two launch syntaxes

```cpp
#include <hip/hip_runtime.h>

__global__ void saxpy(float a, const float* __restrict__ x,
                      float* __restrict__ y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a * x[i] + y[i];   // boundary guard
}

void launch(float a, const float* x, float* y, int n, hipStream_t s) {
    dim3 block(256);
    dim3 grid((n + block.x - 1) / block.x);

    // Triple-chevron form (CUDA-compatible)
    saxpy<<<grid, block, 0, s>>>(a, x, y, n);

    // Macro form — identical semantics, explicit dynamic-LDS + stream args
    hipLaunchKernelGGL(saxpy, grid, block, /*sharedBytes=*/0, s, a, x, y, n);
}
```

Both forms compile to the same dispatch. `hipLaunchKernelGGL(kernel, grid, block,
dynamicSharedBytes, stream, args...)` is preferred in code that must also build
with older toolchains or non-`nvcc`-style host compilers, because it is an
ordinary function-like macro. `__global__` functions must return `void`;
device-only helpers are marked `__device__` (and may be `__host__ __device__` to
share code with the CPU). Check launch errors with `hipGetLastError()` — a bad
grid/block silently no-ops otherwise.

## LDS: `__shared__` and `__syncthreads`

`__shared__` allocates **LDS** (Local Data Share), the per-CU scratchpad that is
AMD's equivalent of CUDA shared memory. It is the staging ground for tiled GEMM,
transposes, and reductions. Capacity is architecture-specific: **64 KiB/CU on
gfx942**, **160 KiB/CU on gfx950** (see [hw-lds](../hardware/lds.md)) — and LDS
usage is a primary occupancy limiter, so size tiles deliberately.

```cpp
template <int TILE>
__global__ void transpose(const float* in, float* out, int w, int h) {
    // +1 is an illustrative correctness-preserving pad, not a universal
    // conflict-free choice; derive it from lane mapping/opcode/target phases.
    __shared__ float tile[TILE][TILE + 1];

    int x = blockIdx.x * TILE + threadIdx.x;
    int y = blockIdx.y * TILE + threadIdx.y;
    if (x < w && y < h)
        tile[threadIdx.y][threadIdx.x] = in[y * w + x];

    __syncthreads();                          // -> s_barrier; all waves in block

    int tx = blockIdx.y * TILE + threadIdx.x;
    int ty = blockIdx.x * TILE + threadIdx.y;
    if (tx < h && ty < w)
        out[ty * h + tx] = tile[threadIdx.x][threadIdx.y];
}
```

For the exact `TILE=32`, `block(32,32)` mapping, gfx942 uses `+1` while gfx950
needs `+2`; see the [worked transpose](../kernels/transpose-lds.md).

`__syncthreads()` lowers to `s_barrier` and synchronizes **all wavefronts in the
block**, not a single wave. The `TILE + 1` column padding skews the row stride so
consecutive rows fall on different LDS banks (gfx942 has 32 banks; gfx950 has 64)
— see [bank-conflict avoidance](../techniques/bank-conflict-avoidance.md).
Dynamic LDS (the `sharedBytes` launch argument) is reached via an
`extern __shared__` array.

## `warpSize` is not 32 — query it

The single most common porting bug. CUDA code assumes a 32-lane warp; on **CDNA
(gfx9xx) every wavefront is 64 lanes** and is wave64-only, while RDNA supports
both. Never hardcode `32`:

```cpp
__global__ void reduce_assumption_safe(const float* in, float* out, int n) {
    // warpSize is a built-in: 64 on gfx942/gfx950, 32 or 64 on gfx1201
    int lane = threadIdx.x % warpSize;
    // __ballot returns a 64-bit mask on AMD — use unsigned long long
    unsigned long long active = __ballot(in[blockIdx.x] > 0.0f);
    if (lane == 0) out[blockIdx.x] = __popcll(active);
}
```

`__ballot`/`__activemask` return **64-bit** masks on AMD. Reductions, ballots,
and lane-stride loops should all derive their span from `warpSize` (or the
compile-time `__AMDGCN_WAVEFRONT_SIZE__`) — see
[hw-wavefront](../hardware/wavefront.md) and
[CUDA→HIP migration](../migration/cuda-to-hip.md).

## AMDGCN builtins

When the portable APIs are not enough, `__builtin_amdgcn_*` exposes the ISA
directly. Three families matter most.

**Matrix cores (MFMA).** A whole wavefront cooperatively computes `D = A·B + C`.
The builtin name encodes output format, shape, and input format:

```cpp
using float4  = __attribute__((__vector_size__(16))) float;
using half4   = __attribute__((__vector_size__(8)))  __fp16;

// 16x16x16 FP16 -> FP32, executed across all 64 lanes
__device__ float4 mma_16x16x16(half4 a, half4 b, float4 c) {
    return __builtin_amdgcn_mfma_f32_16x16x16f16(a, b, c,
                                                 /*cbsz=*/0, /*abid=*/0, /*blgp=*/0);
}
```

Operand-to-register layout is shape-specific; prefer
[rocWMMA](rocwmma.md) or [Composable Kernel](composable-kernel.md), and consult
[hw-mfma](../hardware/mfma.md) before hand-rolling builtins.

**Cross-lane permute.** `__builtin_amdgcn_ds_bpermute` is a backward 64-lane
gather through the LDS crossbar (no LDS storage consumed); the address operand is
`lane_index * 4` in bytes. It is the standard tool for cross-row wavefront
reductions:

```cpp
// Move lane (laneId ^ 32)'s value into this lane — half-wave shuffle
__device__ float xor_shuffle32(float v) {
    int   src  = (threadIdx.x ^ 32) & 63;
    int   data = __builtin_bit_cast(int, v);
    int   r    = __builtin_amdgcn_ds_bpermute(src * 4, data);
    return __builtin_bit_cast(float, r);
}
```

Related lane ops: `__builtin_amdgcn_ds_permute` (forward scatter),
`__builtin_amdgcn_mov_dpp` (row shift/broadcast), and
`__builtin_amdgcn_readlane`/`readfirstlane`. Note
`v_permlane16_swap_b32` / `v_permlane32_swap_b32` are **gfx950-only**; the RDNA
selector form is different. See [hw-cross-lane](../hardware/cross-lane.md) and the
[wave-reduce technique](../techniques/wave-reduce.md).

**Scheduler control.** `__builtin_amdgcn_sched_barrier(mask)` constrains how the
backend reorders instructions across a point — essential for keeping
software-pipelined GEMM (global loads ↔ MFMA ↔ LDS) interleaved the way you laid
it out. `mask = 0` is a hard barrier (no reordering across it); other bits allow
specific instruction classes through. The related
`__builtin_amdgcn_s_barrier` is the raw block barrier behind `__syncthreads`.

```cpp
// Pin the boundary between a prefetch and the compute that consumes it
load_next_tile_to_lds(...);
__builtin_amdgcn_sched_barrier(0);   // compiler may not hoist MFMA above this
compute_on_current_tile(...);
```

## Compiling

```bash
# Multi-target fat binary for CDNA3, CDNA4, and RDNA4
hipcc -O3 --offload-arch=gfx942 --offload-arch=gfx950 \
      --offload-arch=gfx1201 saxpy.hip -o saxpy

# Inspect the generated ISA (verify MFMA / ds_bpermute actually emitted)
hipcc -O3 --offload-arch=gfx942 -S -o - saxpy.hip | less
```

Use `--save-temps` or `rocprofv3`/`rocprof-compute` to confirm the intended
instructions and to read occupancy, VGPR/AGPR, and LDS allocation.

## Gotchas

- **`warpSize == 64`** on CDNA — 64-bit ballot masks; never assume 32.
- `__global__` must return `void`; pass results through pointers.
- `__syncthreads` is a *block* barrier (`s_barrier`); it does not order async
  memory — use `s_waitcnt` semantics for that (see [hw-s-waitcnt](../hardware/s-waitcnt.md)).
- MFMA/DPP/permute builtins are **architecture-specific**; guard with
  `__gfx942__`, `__gfx950__`, `#if __AMDGCN_WAVEFRONT_SIZE__ == 64`, etc.
- Bitcast (`__builtin_bit_cast`) floats to int before `ds_bpermute`; the builtin
  operates on 32-bit lane data.

## See also

- [MFMA matrix-core instructions](../hardware/mfma.md)
- [Wavefront / EXEC / occupancy](../hardware/wavefront.md)
- [Local Data Share](../hardware/lds.md)
- [Cross-lane: DPP, swizzle, permute](../hardware/cross-lane.md)
- [CUDA → HIP migration](../migration/cuda-to-hip.md)

## Sources

- [HIP Programming Guide — Hardware Features](https://rocm.docs.amd.com/projects/HIP/en/latest/)
- [LLVM AMDGPU Backend / User Guide](https://llvm.org/docs/AMDGPUUsage.html)
- [AMD Matrix Cores (ROCm Blogs)](https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores/README.html)
- [Matrix Core Programming on CDNA3 and CDNA4](https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores-cdna/README.html)
