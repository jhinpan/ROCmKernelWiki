---
id: migration-cuda-to-hip
title: CUDA → HIP Kernel Porting (CDNA3/CDNA4)
type: migration
architectures:
- gfx942
- gfx950
tags:
- cdna
- wave64
- mfma
- async-copy
- direct-to-lds
- s-waitcnt
- permute
- dpp
- fp8
- hip
confidence: source-reported
reproducibility: snippet
languages:
- hip
- cpp
related:
- hw-async-copy-lds
- hw-s-waitcnt
- hw-mfma
- migration-wmma-vs-mfma
- hw-wavefront
- lang-hip
sources:
- doc-rocm-hip-hw
- doc-cdna3-isa
- doc-cdna4-isa
- blog-matrix-cores-cdna
- doc-llvm-amdgpu
cross_vendor_note: 'NVIDIA SM and AMD CDNA differ in three ways that break a naive
  1:1 port: (1) warp = 32 lanes on NVIDIA, but a CDNA wavefront is 64 lanes (wave64-only)
  — `__ballot`/`__activemask` return 64-bit masks and per-warp reductions change trip
  count. (2) NVIDIA exposes a named async object model (`cp.async` + `mbarrier`, and
  on Hopper TMA + `wgmma`); AMD has no mbarrier and no TMA — the closest async primitive
  is direct-to-LDS load gated by the global, compiler-scheduled `s_waitcnt` counters.
  (3) Tensor cores: `wgmma`/`wmma` (PTX) map to CDNA `v_mfma_*`, which run per-wavefront
  over 64 lanes with a different register layout, so PTX tile-fragment code must be
  regenerated, not textually translated.'
implemented_by:
- pr-aiter-2136
- pr-composable_kernel-2466
- pr-aiter-3072
- pr-aiter-2394
- pr-Tensile-1406
- pr-Tensile-1288
- pr-composable_kernel-2722
- pr-composable_kernel-2704
---
# CUDA → HIP Kernel Porting (CDNA3/CDNA4)

## Overview

Most CUDA host code ports to HIP almost mechanically: `hipify-perl` / `hipify-clang`
rewrite the runtime/driver API (`cudaMalloc` → `hipMalloc`, `cudaMemcpyAsync` →
`hipMemcpyAsync`, `<<<>>>` is accepted verbatim by `hipcc`). The hard part is the
**device kernel body** — anything that touches the warp width, async memory
pipeline, tensor cores, or cross-lane shuffles needs a real port, not a textual
substitution. This page focuses on those device-side hazards for CDNA3 (gfx942)
and CDNA4 (gfx950).

> Rule of thumb: if a line of CUDA mentions `32`, `cp.async`, `mbarrier`,
> `wgmma`/`wmma`, `__shfl`, or `cooperative_groups`, stop and port it
> deliberately.

## API surface mapping

| CUDA | HIP | Notes |
|---|---|---|
| `cudaMalloc` / `cudaFree` | `hipMalloc` / `hipFree` | mechanical |
| `cudaMemcpyAsync` | `hipMemcpyAsync` | mechanical |
| `__syncthreads()` | `__syncthreads()` | lowers to `s_barrier` |
| `__shared__` | `__shared__` | lowers to LDS allocation |
| `__shfl_sync(mask, v, src)` | `__shfl(v, src, width)` | **mask dropped; width default = warpSize = 64** |
| `__ballot_sync(mask, p)` | `__ballot(p)` | returns **`uint64_t`** on CDNA |
| `wmma::*` fragments (PTX) | `rocwmma::*` / `__builtin_amdgcn_mfma_*` | regenerate, see below |
| `cp.async` + `cp.async.wait_group` | `__builtin_amdgcn_load_to_lds` + `s_waitcnt` | direct-to-LDS, see below |
| `__nanosleep` | `__builtin_amdgcn_s_sleep` | coarse granularity |

## warpSize: 32 → 64

CDNA is **wave64-only**. The single most common porting bug is a hard-coded `32`.
Query the width; do not assume it:

```cpp
// WRONG on CDNA: assumes 32-lane warps
// int lane = threadIdx.x % 32;

// PORTABLE: warpSize is 64 on gfx9xx, 32 on gfx10+/RDNA
const int lane = threadIdx.x % warpSize;   // warpSize is a builtin
const int wid  = threadIdx.x / warpSize;

// Full-wave mask is 64-bit on AMD:
unsigned long long active = __ballot(pred); // NOT unsigned (32-bit)
int popc = __popcll(active);
```

A block of 256 threads is **4 wavefronts** on CDNA, not 8 warps. Loop trip counts
for warp-level reductions, shared-memory staging sizes per warp, and
`blockDim`-derived indexing all shift accordingly.

## `__shfl`: warp shuffles

HIP's `__shfl*` builtins are unmasked and default to `width = warpSize`. Under the
hood they lower to `ds_bpermute_b32` / DPP on CDNA. A typical CUDA warp reduction
ports like this:

```cpp
// CUDA: for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffff, v, o);
// HIP on CDNA (64 lanes -> start at 32):
#pragma unroll
for (int o = warpSize / 2; o > 0; o >>= 1)
    v += __shfl_down(v, o);              // width defaults to warpSize (64)
```

For the lowest-latency cross-lane patterns on AMD, prefer the native intrinsics
(`__builtin_amdgcn_ds_bpermute`, `__builtin_amdgcn_mov_dpp`,
`__builtin_amdgcn_readlane`) — see the [cross-lane page](../hardware/cross-lane.md).
Note `v_permlane16_*` exists only on gfx950, not gfx942.

## `cp.async` → direct-to-LDS load

NVIDIA's `cp.async` streams global memory into shared memory bypassing registers.
AMD's analog is the **direct-to-LDS** load: `buffer_load_dword ... lds` /
`global_load_lds_*`, which moves HBM → LDS without occupying VGPRs. There is **no
async *group* object** — completion is tracked by the `vmcnt` counter, not a
per-copy handle.

```cpp
// CUDA cp.async (sketch):
//   __pipeline_memcpy_async(&smem[i], &gmem[i], 16);
//   __pipeline_commit(); __pipeline_wait_prior(0);

// HIP / CDNA direct-to-LDS via the LLVM intrinsic:
//   copies 4 bytes/lane HBM -> LDS, no VGPR staging.
__builtin_amdgcn_load_to_lds(
    /*src global ptr*/ gptr + offset,
    /*dst LDS ptr  */ sptr,
    /*size bytes   */ 4,          // gfx950 also allows 12/16 (dwordx3/x4)
    /*offset       */ 0,
    /*aux          */ 0);

// ... issue more loads, then do independent compute ...

__builtin_amdgcn_s_waitcnt(/*vmcnt=*/0);  // wait for the LDS fills to land
__syncthreads();                          // then publish to the whole block
```

This frees the ArchVGPRs that a load-then-`ds_write` path would burn, which
directly buys occupancy in GEMM/attention mainloops. Details and the gfx950
widening to 12/16-byte transfers are on the
[direct-to-LDS page](../hardware/async-copy-lds.md).

## `mbarrier` / `cp.async.wait` → `s_waitcnt`

CUDA's `mbarrier` is a named, addressable barrier object with a phase and an
arrive/wait protocol; Hopper pairs it with TMA for bulk-async transfers. **CDNA
has neither.** Instead, every wavefront carries three monotonic hardware counters
that the compiler schedules:

| Counter | Tracks | Replaces |
|---|---|---|
| `vmcnt` | outstanding VMEM (buffer/flat/global) loads & stores | `cp.async.wait_group`, TMA wait |
| `lgkmcnt` | LDS + scalar-const + GDS/message ops | shared-memory dependency waits |
| `expcnt` | export/GDS (CDNA3; unused on CDNA4) | — |

`s_waitcnt vmcnt(N)` blocks until all but `N` outstanding VMEM ops retire, which
is how you express "wait for all-but-the-last-stage loads" in a software pipeline:

```cpp
// double-buffered mainloop skeleton
issue_loads(stage ^ 1);                 // prefetch next tile to LDS
__builtin_amdgcn_s_waitcnt(0x0f70);     // vmcnt(0): wait for THIS stage's loads
__syncthreads();
mfma_compute(stage);                    // consume current tile
stage ^= 1;
```

Because same-type ops complete **in order** but different types may complete out
of order, you generally cannot replace a fine-grained mbarrier phase with a
single counter wait — you restructure the loop so the counter ordering provides
the dependency. See the [s_waitcnt page](../hardware/s-waitcnt.md).

## `wgmma` / `wmma` → `mfma`

PTX tensor-core instructions do not survive a textual port. On CDNA, the
matrix-core op is `v_mfma_*`, issued cooperatively across the **whole 64-lane
wavefront**, with a register layout that differs from NVIDIA's per-warp fragment
layout. Concretely:

```cpp
// CUDA (PTX): wmma.mma.sync.aligned.m16n16k16.f32.f16.f16.f32 ...
// HIP/CDNA equivalent tile: 16x16x16 FP16 -> FP32 accumulate
using float4  = __attribute__((__vector_size__(16))) float;
using half4   = __attribute__((__vector_size__(8)))  __fp16;

__device__ float4 mma_16x16x16(half4 a, half4 b, float4 c) {
    return __builtin_amdgcn_mfma_f32_16x16x16f16(a, b, c, 0, 0, 0);
}
```

Practical guidance:

- **Don't hand-translate.** Use [rocWMMA](../languages/rocwmma.md) (a fragment API
  intentionally close to `nvcuda::wmma`) so the same source compiles to `wmma`
  on NVIDIA and `v_mfma_*` on CDNA, or let Composable Kernel / hipBLASLt / Triton
  emit MFMA.
- **Pick a real shape.** gfx942 dense FP16 shapes are `16x16x16` and `32x32x8`;
  FP8 is `16x16x32` / `32x32x16`. gfx950 adds the unified `f8f6f4` path
  (`16x16x128`, `32x32x64`) and MX-scaled variants. See [MFMA](../hardware/mfma.md).
- **FP8 is not portable bit-for-bit.** gfx942 uses **FNUZ** FP8; gfx950 and NVIDIA
  use **OCP** FP8. Reusing quantized weights across these requires reinterpreting
  the encoding — see [gfx942 → gfx950](gfx942-to-gfx950.md) and
  [WMMA vs MFMA](wmma-vs-mfma.md).
- RDNA4 (gfx1201) uses **WMMA**, not MFMA, and supports both wave32 and wave64 —
  another reason to query `warpSize` rather than branch on vendor.

## Porting checklist

1. Run `hipify-clang` on host/runtime code; review the device kernels by hand.
2. Replace every literal `32` lane-width with `warpSize`; widen masks to 64-bit.
3. Convert `__*_sync(mask, …)` shuffles/ballots to the unmasked HIP builtins.
4. Replace `cp.async` + `mbarrier` with direct-to-LDS loads + `s_waitcnt vmcnt`.
5. Regenerate tensor-core tiles via rocWMMA / a library; never translate PTX MMA.
6. Re-tune block size (256 threads = 4 waves), LDS budget (64 KB gfx942 / 160 KB
   gfx950), and VGPR/AGPR pressure for occupancy.

## See also

- [Direct-to-LDS async copy](../hardware/async-copy-lds.md)
- [s_waitcnt counters](../hardware/s-waitcnt.md)
- [MFMA matrix cores](../hardware/mfma.md)
- [WMMA vs MFMA](wmma-vs-mfma.md)
- [Wavefront / occupancy](../hardware/wavefront.md)

## Sources

- [HIP Programming — Hardware Features & warpSize](https://rocm.docs.amd.com/projects/HIP/en/latest/)
- [CDNA3 ISA Reference Guide](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf)
- [CDNA4 ISA Reference Guide](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-cdna4-instruction-set-architecture.pdf)
- [Matrix Core Programming on CDNA3 and CDNA4](https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores-cdna/README.html)
- [LLVM AMDGPU Backend — User Guide](https://llvm.org/docs/AMDGPUUsage.html)
