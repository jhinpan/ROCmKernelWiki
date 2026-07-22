---
id: hw-async-copy-lds
title: Direct-to-LDS Async Copy — buffer_load…lds / global_load_lds
type: hardware
architectures:
- gfx942
- gfx950
tags:
- async-copy
- direct-to-lds
- lds-bypass
- lds
- s-waitcnt
- buffer-instructions
- global-instructions
- vgpr
confidence: verified
evidence_basis:
- source_id: doc-cdna3-isa
  evidence_type: official-doc
- source_id: doc-cdna4-isa
  evidence_type: official-doc
- source_id: doc-llvm-amdgpu
  evidence_type: official-doc
- source_id: ref-gcnasm
  evidence_type: upstream-code
related:
- hw-s-waitcnt
- hw-lds
- hw-memory-instructions
- technique-lds-double-buffering
- migration-cuda-to-hip
sources:
- doc-cdna3-isa
- doc-cdna4-isa
- doc-llvm-amdgpu
- ref-gcnasm
cross_vendor_note: 'This is AMD''s closest analog to NVIDIA''s cp.async (Ampere) —
  an HBM→shared-memory copy that bypasses the register file. There is no TMA/mbarrier
  equivalent on CDNA; completion is gated by the generic VMCNT counter, not a dedicated
  barrier object.

  '
aliases:
- direct-to-LDS
- direct to LDS
- async copy
- global_load_lds
- buffer_load_dword lds
- cp.async analog
implemented_by:
- pr-composable_kernel-2949
- pr-composable_kernel-2573
- pr-composable_kernel-2425
- pr-FlyDSL-139
- pr-aiter-2394
- pr-composable_kernel-2984
- pr-composable_kernel-2545
- pr-aiter-3336
---
# Direct-to-LDS Async Copy — `buffer_load…lds` / `global_load_lds`

## Overview

The normal way to stage data into LDS (shared memory) on CDNA is a two-step
round trip: a VMEM load (`buffer_load`/`global_load`) lands the data in VGPRs,
then a `ds_write` copies it from VGPRs into LDS. The **direct-to-LDS** path
collapses this into a single instruction: the memory unit streams data straight
from HBM/L2 into LDS, **bypassing the vector register file entirely**.

This is the AMD analog of NVIDIA's `cp.async`. It buys two things that matter for
GEMM and attention prologue/mainloop pipelines:

- **VGPR pressure relief.** The staging registers disappear, so the freed
  ArchVGPRs raise occupancy or allow larger accumulator tiles (see
  [VGPR budgeting](../patterns/vgpr-pressure.md)).
- **Overlap.** The copy is asynchronous — issue it, do MFMA work on the previous
  tile, then drain with `s_waitcnt vmcnt(...)`. This is the backbone of
  [LDS double buffering](../techniques/lds-double-buffering.md).

## The instructions

Two MUBUF/FLAT variants carry the `lds` bit:

- **`buffer_load_dword … lds`** (MUBUF) — uses a 128-bit buffer resource
  descriptor (V#). Its source address is range-checked, but do not assume that
  an OOB direct-to-LDS transfer writes zero into LDS: that destination behavior
  is distinct from an ordinary MUBUF-to-VGPR load and must be measured for the
  target. The per-lane source address comes from the V# plus the lane offset;
  the LDS destination offset comes from `M0`.
- **`global_load_lds_dword` / `…_dwordx{3,4}`** (FLAT global) — flat 64-bit
  addressing (SADDR + VGPR offset), no descriptor needed.

In both cases the data never touches a VGPR. The LDS write address is derived
from the `M0` register, so **`M0` must be initialized** to the LDS base/offset
before issuing the copy. Completion is tracked by **VMCNT** (it is a VMEM op),
*not* LGKMCNT — even though the destination is LDS — so you drain it like any
other global load.

> **Width limits.** `__builtin_amdgcn_load_to_lds` accepts byte sizes
> **1, 2, and 4** on gfx942. gfx950 accepts **1, 2, 4, 12, and 16**; its wider
> forms lower to `global_load_lds_dwordx3` / `global_load_lds_dwordx4`.

## Emitting it from HIP / LLVM

Clang exposes the copy through the `llvm.amdgcn.load.to.lds` intrinsic
(builtin `__builtin_amdgcn_load_to_lds`). The compiler selects the right
`buffer`/`global` `…lds` instruction and manages `M0`.

```cpp
#include <hip/hip_runtime.h>

// Stage a tile of `global` (HBM) into `__shared__` LDS without touching VGPRs.
// The LDS destination passed to the intrinsic must be wave-uniform. Hardware
// applies the per-lane destination offset (4*lane for this 4-byte copy).
__global__ void prefetch_tile(const float* __restrict__ g_in, float* out, int n)
{
    __shared__ float tile[256];

    const int lane = threadIdx.x % warpSize;
    const int wave = threadIdx.x / warpSize;
    const float* src = g_in + blockIdx.x * 256 + threadIdx.x;
    float* wave_uniform_dst = &tile[wave * warpSize];
    __builtin_amdgcn_load_to_lds(
        /*src  global ptr */ src,
        /*wave-uniform LDS base*/ wave_uniform_dst,
        /*size bytes      */ sizeof(float),
        /*offset          */ 0,
        /*aux (cache ctrl)*/ 0);

    // The copy is async and counted by VMCNT. Drain before reading LDS.
    __builtin_amdgcn_s_waitcnt(0);               // or vmcnt(0) via the encoded imm
    __syncthreads();                             // s_barrier: tile visible to all

    if (blockIdx.x * 256 + lane < n)
        out[blockIdx.x * 256 + lane] = tile[lane] * 2.0f;
}
```

The same path is what the Triton AMD backend and Composable Kernel emit for
their software-pipelined mainloops on gfx950.

## What it looks like in assembly

The hand-written form (the way you would see it in a Tensile / `gcnasm`
mainloop) makes the VGPR-bypass explicit — there is no `ds_write` after the
load:

```asm
; ---- prologue: point M0 at the LDS destination base ----
        s_mov_b32       m0, s_lds_base          ; LDS byte offset for the write

; ---- gfx942: 4 bytes/lane, descriptor-based ----
        buffer_load_dword  v_addr, s[0:3], 0 offen lds   ; HBM -> LDS, no VGPR dst

; ---- gfx950: 16 bytes/lane in one shot, flat global addressing ----
        global_load_lds_dwordx4  v_addr, s[8:9] offset:0 ; 16B HBM -> LDS

; ... issue several copies, then do MFMA on the previous buffer ...
        s_waitcnt       vmcnt(0)                ; wait for direct-to-LDS to land
        s_barrier                               ; make LDS visible wave-wide
        ds_read_b128    v[0:3], v_lds_off       ; now consume the staged tile
```

Note that the `lds` MUBUF/FLAT loads have **no VGPR data destination operand** —
the address VGPR is the only vector operand; the payload goes directly to LDS at
`M0 + per-lane offset`.

## Pipelining pattern

The intended use is to keep `N` copies in flight while computing on already-staged
tiles, then partially drain:

1. Issue direct-to-LDS copies for tile *k+1* (and *k+2* for deeper pipelines).
2. Run `v_mfma_*` on tile *k* (already in LDS / VGPRs).
3. `s_waitcnt vmcnt(N)` — block only until all but the freshest `N` copies retire.
4. `s_barrier`, swap buffers, repeat.

Because completion is just VMCNT, the compiler schedules these as ordinary VMEM
ops; there is no `mbarrier::arrive`/`wait` handshake to manage. See
[s_waitcnt](s-waitcnt.md) for the counter semantics and
[LDS double buffering](../techniques/lds-double-buffering.md) for the full loop.

## Caveats

- **`M0` is shared state.** It also drives `ds_*` indexed ops and message
  instructions; if you interleave those with direct-to-LDS copies you must
  re-establish `M0` before each consumer.
- **No VGPR transform on the fly.** Because the data never enters a VGPR, you
  cannot pack/convert during the copy — staging into LDS, then `ds_read` + VALU,
  is still required for layout changes (e.g. transpose, dtype repack).
- **Alignment.** The widened gfx950 `dwordx3/x4` forms expect naturally aligned
  addresses; misalignment forces the assembler/compiler back to narrower
  transfers and costs the instruction-count win.
- **Bank conflicts still apply** on the eventual `ds_read` of the staged tile —
  direct-to-LDS only removes the write-side VGPR round trip, not LDS bank
  arbitration. See [LDS](lds.md).

## See also

- [s_waitcnt — async gating](s-waitcnt.md)
- [Local Data Share (LDS)](lds.md)
- [Memory instructions: buffer vs global vs flat](memory-instructions.md)
- [LDS double buffering technique](../techniques/lds-double-buffering.md)
- [CUDA → HIP migration (cp.async → direct-to-LDS)](../migration/cuda-to-hip.md)

## Sources

- [AMD Instinct MI300 / CDNA3 ISA Reference Guide](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf)
- [AMD Instinct CDNA4 ISA Reference Guide](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-cdna4-instruction-set-architecture.pdf)
- [LLVM AMDGPU Backend User Guide (`llvm.amdgcn.load.to.lds`)](https://llvm.org/docs/AMDGPUUsage.html)
- [AMD GCN/CDNA assembly examples (gcnasm)](https://github.com/AMD-AI/amd_gcn_assembly)
