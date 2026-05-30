---
id: technique-lds-double-buffering
title: "LDS Double / Multi-Buffering (Overlapping HBM Loads with MFMA)"
type: technique
architectures:
- gfx942
- gfx950
tags:
- lds-double-buffering
- software-pipelining
- direct-to-lds
- async-pipeline
- lds
- async-copy
- s-waitcnt
- mfma
confidence: source-reported
reproducibility: snippet
hardware_features:
- lds
- async-copy
- s-waitcnt
- mfma
- ds-instructions
kernel_types:
- gemm
- hgemm
- fp8-gemm
- flash-attention
languages:
- hip
- gcn-asm
related:
- hw-async-copy-lds
- hw-lds
- hw-s-waitcnt
- technique-mfma-pipelining
- technique-lds-swizzling
- pattern-memory-bound
sources:
- hw-async-copy-lds
- hw-lds
- hw-s-waitcnt
- blog-gemm-optimization
- ref-gcnasm
---

# LDS Double / Multi-Buffering

## Overview

In a tiled GEMM (or attention) inner loop, each main-loop iteration must (1) load
the next K-tile of A and B from HBM into LDS, and (2) feed the *current* K-tile
from LDS into the matrix cores via `v_mfma_*`. If these two phases run serially —
load, `s_barrier`, compute, repeat — the matrix unit stalls on every iteration
waiting for global memory latency (hundreds of cycles on MI300X). The result is a
**memory-latency-bound** loop whose MFMA utilization is far below the
[peak FLOP figures](../hardware/mfma.md).

**Double buffering** breaks that dependency by allocating two LDS tiles (`buf[0]`,
`buf[1]`) and pipelining one iteration ahead: while the matrix cores consume
`buf[i % 2]`, the DMA path streams the *next* K-tile into `buf[(i+1) % 2]`. The
load latency of iteration *i+1* is hidden underneath the compute of iteration *i*.
This is AMD's equivalent of the classic NVIDIA `cp.async` + `commit/wait_group`
software pipeline, built instead from **direct-to-LDS loads** and **`s_waitcnt
vmcnt(N)`** counter gating.

Two hardware features make this efficient on CDNA:

- [**Direct-to-LDS (async copy)**](../hardware/async-copy-lds.md):
  `buffer_load_dword ... lds` / `global_load_lds_*` copy HBM→LDS while **bypassing
  the VGPR file**. This both frees VGPRs (raising occupancy) and lets the load
  stream proceed concurrently with VALU/MFMA work.
- [**`s_waitcnt vmcnt(N)`**](../hardware/s-waitcnt.md): the wave's outstanding-VMEM
  counter lets you issue many loads and then block only until *all but N* of them
  retire — the gate that lets next-tile loads stay in flight across the compute of
  the current tile.

## Why it works: the counter, not a barrier

Unlike NVIDIA's `mbarrier`/`cp.async.wait_group`, AMD has **no named async-copy
barrier object**. Instead each wave carries a single monotonic `VMCNT` (6-bit)
that counts outstanding VMEM operations and decrements as data lands. The compiler
(or hand-written asm) schedules loads early and inserts a `s_waitcnt vmcnt(N)`
right before the data is actually needed. Same-type ops complete **in order**, so
`vmcnt(N)` deterministically means "the N most-recently-issued loads may still be
in flight." LDS reads/writes are gated separately by `LGKMCNT`.

The canonical double-buffered schedule per iteration is therefore:

```
issue async loads for tile (i+1)   ; buffer_load ... lds  -> buf[(i+1)&1]
s_waitcnt vmcnt(0)                  ; ensure tile (i) loads (issued last iter) are in LDS
s_barrier                          ; all waves see buf[i&1] populated
ds_read   buf[i&1] -> VGPR/AGPR     ; stage operands
v_mfma_*  ... (consume tile i)      ; compute overlaps with tile (i+1) DMA in flight
```

By keeping the tile-(i+1) loads issued *before* the `vmcnt`/`barrier`/compute of
tile *i*, their latency is paid down during the MFMA work.

## HIP implementation sketch

The portable way to emit direct-to-LDS copies from HIP is the
`__builtin_amdgcn_load_to_lds` intrinsic (LLVM `llvm.amdgcn.load.to.lds`). The
compiler lowers it to `buffer_load_*`/`global_load_lds_*` and tracks `vmcnt` for
you when you place `__builtin_amdgcn_s_waitcnt`-equivalent fences via
`__builtin_amdgcn_s_barrier` and the wait builtins.

```cpp
#include <hip/hip_runtime.h>

constexpr int BK   = 32;            // K-tile depth
constexpr int TILE = 128 * BK;      // elements of A (and B) per K-tile
constexpr int NBUF = 2;             // double buffer

// Stream one 4-byte dword per lane from global -> LDS, bypassing VGPRs.
__device__ inline void cp_async_lds(__attribute__((address_space(3))) float* dst,
                                    const float* src) {
  // size_in_bytes = 4 (dword). gfx950 also supports 12/16-byte widths.
  __builtin_amdgcn_load_to_lds(src, dst, /*size=*/4, /*offset=*/0, /*aux=*/0);
}

__global__ void gemm_db(const float* __restrict__ A,
                        const float* __restrict__ B,
                        float* __restrict__ C, int K) {
  __shared__ float a_lds[NBUF][TILE];
  __shared__ float b_lds[NBUF][TILE];

  const int lane = threadIdx.x;
  int buf = 0;

  // ---- Prologue: kick off the first tile's loads (no compute yet) ----
  cp_async_lds(&a_lds[buf][lane], &A[lane]);
  cp_async_lds(&b_lds[buf][lane], &B[lane]);

  acc_t acc = {};                                   // AGPR accumulator tile
  for (int k0 = 0; k0 < K; k0 += BK) {
    const int nbuf = buf ^ 1;

    // 1) Prefetch NEXT tile into the other buffer (stays in flight).
    if (k0 + BK < K) {
      cp_async_lds(&a_lds[nbuf][lane], &A[(k0 + BK) * 128 + lane]);
      cp_async_lds(&b_lds[nbuf][lane], &B[(k0 + BK) * 128 + lane]);
    }

    // 2) Wait only for THIS tile's loads (vmcnt leaves next-tile loads pending).
    __builtin_amdgcn_s_barrier();                   // + compiler-inserted s_waitcnt vmcnt

    // 3) Consume current tile on the matrix cores.
    mfma_ktile(acc, a_lds[buf], b_lds[buf]);        // ds_read -> v_mfma_f32_16x16x16_f16 ...

    buf = nbuf;                                      // flip
  }
  store_acc(C, acc);
}
```

> The `__builtin_amdgcn_load_to_lds` call requires LLVM/ROCm that exposes the
> intrinsic; `acc_t`, `mfma_ktile`, and `store_acc` stand in for your MFMA tiling
> (see [MFMA pipelining](mfma-pipelining.md)). The key structural point is that
> the **next-tile prefetch is issued before the barrier/compute of the current
> tile**, so its latency overlaps.

## Hand-asm form (the gate that matters)

In Tensile/CK-style assembly the same idea is explicit — issue all the loads for
the prefetch buffer, then drain with a `vmcnt`:

```asm
; prefetch next K-tile A/B directly into LDS (bypasses VGPRs)
buffer_load_dword v[off], s[desc:desc+3], 0 offen lds
buffer_load_dword v[off], s[descB:descB+3], 0 offen lds
; ... rest of MFMA macro-tile for the CURRENT buffer issues here ...
s_waitcnt vmcnt(0)        ; current-tile loads have landed in LDS
s_barrier                 ; cross-wave sync before ds_read
ds_read_b128 a[0:3], v[ldsAddr]
v_mfma_f32_16x16x16_f16 a[0:3], v[0:1], v[2:3], a[0:3]
```

## Buffer count, LDS budget, and occupancy

Double buffering costs **2×** the LDS of a single tile; *N*-way (multi-) buffering
costs *N*×. LDS is a hard occupancy limiter, so the tile depth `BK` and buffer
count trade directly against waves/CU:

| Arch | LDS/CU | Banks | Practical budget for buffers |
|---|---|---|---|
| gfx942 (CDNA3) | 64 KB | 32 × 512 dwords | tight — usually 2 buffers |
| gfx950 (CDNA4) | 160 KB | 64 × 640 dwords | room for 2–3+ buffers / deeper K |

The larger LDS on [CDNA4](../hardware/lds.md) is what makes deeper multi-stage
pipelines (3+ buffers, à la a longer software pipeline) practical without
collapsing occupancy. On gfx942 you typically pick `BK` so two buffers fit
alongside the AGPR accumulator tile.

Two further interactions to keep in mind:

- **Bank conflicts.** The doubled LDS footprint does not change per-access bank
  behavior, but the staging `ds_read`s still need a conflict-free layout — pad or
  swizzle the tile (see [LDS swizzling](lds-swizzling.md) and
  [bank-conflict avoidance](bank-conflict-avoidance.md)).
- **VGPR savings from direct-to-LDS.** Because the copy bypasses VGPRs, you reclaim
  the registers a load→store-through-VGPR path would have needed, often buying back
  an extra wave or two of occupancy on top of the latency hiding.

## When to use it

Reach for LDS double-buffering whenever a tiled kernel is
[memory-latency-bound](../patterns/memory-bound.md) with the matrix cores idling
between K-tiles — dense GEMM, FP8 GEMM, and FlashAttention K/V streaming are the
prime cases. If the loop is already compute-bound (MFMA-saturated), additional
buffering only spends LDS without payoff; tune the
[MFMA pipeline](mfma-pipelining.md) and occupancy instead.

## Sources

- [CDNA3 ISA Reference Guide — direct-to-LDS loads & `s_waitcnt`](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf)
- [CDNA4 ISA Reference Guide — widened `global_load_lds`](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-cdna4-instruction-set-architecture.pdf)
- [AMD ROCm Blog — Optimizing GEMM kernels (LDS prefetch / pipelining)](https://rocm.blogs.amd.com/artificial-intelligence/matrix-cores/README.html)
- [AMDGPU assembly notes — `buffer_load ... lds`, `s_waitcnt vmcnt` (gcnasm)](https://github.com/RadeonOpenCompute/ROCm)
