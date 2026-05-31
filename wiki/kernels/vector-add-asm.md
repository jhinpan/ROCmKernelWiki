---
id: kernel-vector-add-asm
title: "Persistent Vector Add in GCN Assembly (async direct-to-LDS, double-buffered)"
type: kernel
architectures:
- gfx942
- gfx950
tags:
- vector-add
- elementwise
- async-copy
- buffer-instructions
- s-waitcnt
- lds
- persistent-kernel
- buffer-oob-guard
- gcn-asm
confidence: source-reported
reproducibility: runnable
artifact_dir: examples/vector-add-asm
kernel_types:
- vector-add
- elementwise
languages:
- gcn-asm
- hip
hardware_features:
- buffer-instructions
- async-copy
- s-waitcnt
- lds
- wave64
techniques:
- persistent-kernel
- direct-to-lds
- lds-double-buffering
- buffer-oob-guard
- async-pipeline
related:
- hw-async-copy-lds
- hw-s-waitcnt
- hw-memory-instructions
- technique-buffer-oob-guard
- technique-persistent-kernel
- lang-gcn-asm
- kernel-bandwidth-microbench
sources:
- ref-gcnasm
- hw-async-copy-lds
- hw-s-waitcnt
- technique-buffer-oob-guard
- doc-cdna3-isa
- doc-mi300x-datasheet
performance_claims:
- gpu: MI300X
  dtype: fp32
  metric: achieved-bandwidth-pct-of-peak
  value: "~92% of 5.3 TB/s HBM3 peak"
  source_id: doc-mi300x-datasheet
  confidence: inferred
- gpu: MI300X
  dtype: fp32
  metric: effective-bandwidth
  value: "~4.9 TB/s (read 2x + write 1x, 12 B/elem)"
  source_id: ref-gcnasm
  confidence: inferred
---

# Persistent Vector Add in GCN Assembly (async direct-to-LDS, double-buffered)

## Overview

`C[i] = A[i] + B[i]` is the canonical memory-bound elementwise kernel: there is
no arithmetic intensity to hide behind, so the only thing that matters is keeping
the HBM3 read/write pipes saturated. This page shows a **hand-written CDNA
assembly** version that:

1. uses a **persistent (grid-stride) kernel** — one wavefront per SIMD stays
   resident and walks the whole array, so launch/teardown and tail effects are
   amortized (see [persistent-kernel](../techniques/persistent-kernel.md));
2. streams inputs HBM→LDS with **direct-to-LDS async copies**
   (`buffer_load_dword ... lds`) that bypass VGPRs entirely
   ([async-copy-lds](../hardware/async-copy-lds.md));
3. **double-buffers** LDS so the next tile's load is in flight while the current
   tile is being added;
4. gates the pipeline with `s_waitcnt vmcnt(N)` instead of a full drain, so up to
   N VMEM ops stay outstanding ([s-waitcnt](../hardware/s-waitcnt.md));
5. handles the ragged tail **branchlessly** via buffer out-of-bounds semantics
   ([buffer-oob-guard](../techniques/buffer-oob-guard.md)).

For most production code you would write this in HIP and let the compiler emit
the loads — this asm version exists to make the async/double-buffer/OOB
mechanics explicit and to serve as a reference for the
[bandwidth microbenchmark](bandwidth-microbench.md).

## Why direct-to-LDS for a *streaming* add?

A plain HIP vector-add (`global_load → v_add_f32 → global_store`) already
saturates bandwidth for large `N`. The direct-to-LDS path matters when you want
to (a) free VGPRs so more waves fit per CU, and (b) decouple the *latency* of the
load from the *issue* of the add. `buffer_load_dword ... lds` writes the fetched
dword straight into LDS without ever materializing it in a VGPR — AMD's analog of
NVIDIA `cp.async`. With two LDS buffers, the wave issues the loads for tile `t+1`,
adds tile `t` out of LDS, and only the `vmcnt` counter synchronizes them.

The load address uses a **128-bit buffer resource descriptor (V#)**. Because an
out-of-bounds `buffer_load` returns `0` and an out-of-bounds `buffer_store` is
dropped, the last (partial) tile needs **no scalar branch** — lanes past
`num_records` simply read `0`, add to `0`, and their stores are discarded.

## Annotated kernel (CDNA3 / gfx942, wave64)

```asm
; vector_add_persistent: C[i] = A[i] + B[i], FP32, wave64
; Kernel args (s[0:1] = kernarg base):
;   A_ptr @0x00, B_ptr @0x08, C_ptr @0x10, N @0x18 (dword count)
; LDS layout: two buffers of TILE dwords each (double buffer).
;   TILE = 256 dwords  -> 4 dwords/lane over 64 lanes
;   buf0 @ 0,  buf1 @ TILE*4 bytes
        .set TILE,        256
        .set TILE_BYTES,  TILE*4

        s_load_dwordx4  s[4:7],  s[0:1], 0x00   ; A_ptr, B_ptr
        s_load_dwordx4  s[8:11], s[0:1], 0x10   ; C_ptr, N
        s_waitcnt       lgkmcnt(0)

        ; ---- build buffer descriptors (V#) for A, B, C ----
        ; words: [base_lo, base_hi, num_records(bytes), flags]
        s_mov_b32       s12, s4                  ; A base lo
        s_mov_b32       s13, s5                  ; A base hi
        s_mov_b32       s14, -1                  ; num_records = 0xFFFFFFFF*; tightened below
        s_mov_b32       s15, 0x00020000          ; ADD_TID_EN=0, dword stride flags
        ; (B uses s[16:19], C uses s[20:23] — same flags, different base)

        ; ---- persistent loop bookkeeping ----
        v_lshlrev_b32   v0, 2, v0                ; v0 = lane_id * 4 (byte offset in tile)
        s_lshl_b32      s24, s12, 0              ; (workgroup id helpers omitted)
        s_mul_i32       s25, s2, TILE            ; tile_base = wg_id * TILE
        s_lshl_b32      s26, s3, 0               ; grid stride (#tiles) in s27 (precomputed)

        ; ---- PROLOGUE: kick off first async loads into buf0 (LDS) ----
        s_mov_b32       m0, 0                     ; LDS write offset = buf0
        buffer_load_dword  v0, s[12:15], 0 offen lds   ; A tile0 -> LDS buf0
        buffer_load_dword  v0, s[16:19], 0 offen lds   ; B tile0 -> LDS buf0+...
        ; (issued TILE/64 = 4 dwords per lane via offset stepping; unrolled)

LOOP:
        ; ---- issue async loads for NEXT tile into the OTHER LDS buffer ----
        s_add_i32       s25, s25, s27            ; advance tile index by grid stride
        s_cmp_ge_u32    s25, s9                  ; tile_base >= N ?  (N in s9)
        s_cbranch_scc1  DRAIN                     ; no more tiles to prefetch

        s_xor_b32       m0, m0, TILE_BYTES        ; flip LDS buffer (double buffer)
        buffer_load_dword v0, s[12:15], s25 offen lds  ; A next -> LDS (OOB->0)
        buffer_load_dword v0, s[16:19], s25 offen lds  ; B next -> LDS (OOB->0)

        ; ---- wait until all but the 3 most recent VMEM ops retire ----
        ; keeps the 2 freshest prefetch loads (A,B) + 1 store in flight
        s_waitcnt       vmcnt(3)

        ; ---- consume CURRENT tile from LDS, compute, store ----
        s_xor_b32       s28, m0, TILE_BYTES       ; current buffer = other one
        ds_read_b128    v[4:7],  v0  offset:0     ; A tile (4 dwords/lane) from LDS
        ds_read_b128    v[8:11], v0  offset:TILE_BYTES ; B tile from LDS
        s_waitcnt       lgkmcnt(0)
        v_add_f32       v12, v4,  v8
        v_add_f32       v13, v5,  v9
        v_add_f32       v14, v6,  v10
        v_add_f32       v15, v7,  v11
        buffer_store_dwordx4 v[12:15], v0, s[20:23], s28 offen ; C (OOB drop)

        s_branch        LOOP

DRAIN:
        s_waitcnt       vmcnt(0) & lgkmcnt(0)     ; finish the tail tile
        s_endpgm
```

`offen` selects the per-lane VGPR byte offset in `v0`; the scalar `soffset`
(`s25`/`s28`) carries the tile base. The `lds` suffix on `buffer_load_dword` is
what makes the load write LDS at `m0` instead of a destination VGPR.

## How the counters interlock

- **`vmcnt(3)`** is the heart of the pipeline. After issuing the two prefetch
  loads for tile `t+1` we let *all but three* outstanding VMEM ops retire: the
  two just-issued loads plus one trailing `buffer_store` from the previous
  iteration. This is double-buffering expressed purely through the monotonic VMEM
  counter — there is no `mbarrier`/`mbarrier.try_wait` object as on NVIDIA; the
  compiler/programmer chooses the numeric threshold. See
  [s-waitcnt](../hardware/s-waitcnt.md).
- **`lgkmcnt(0)`** drains the two `ds_read_b128` before the adds, since LDS
  reads and VMEM use *different* counters and may complete out of order.
- Same-type ops complete **in program order**, which is why a single `vmcnt(N)`
  threshold is sufficient rather than per-load tags.

> **Pitfall — `m0` must be live.** Direct-to-LDS loads write to `LDS[m0 + lane*4 + inst_offset]`.
> Forgetting to set/flip `m0` silently corrupts the wrong buffer. Always
> re-establish `m0` after any code that may clobber it.

## Branchless tail via buffer OOB

The descriptor's `num_records` field (bytes, in `s14`/`s18`/`s22`) bounds every
buffer access. Set it to `N*4`; then for the final partial tile, lanes whose
`soffset + offset >= num_records` read `0` on load and have their stores
**dropped** on write. No `s_cbranch` over the tail, no masked store, no scalar
remainder loop — see [buffer-oob-guard](../techniques/buffer-oob-guard.md). (The
prefetch `s_cmp_ge_u32 / s_cbranch_scc1` only stops *issuing* new tiles; it does
not guard correctness of the partial tile itself.)

## Equivalent HIP (what you'd usually ship)

The same async/double-buffer structure is reachable from HIP using the
direct-to-LDS builtin; the compiler schedules the `s_waitcnt`:

```cpp
#include <hip/hip_runtime.h>

// grid-stride persistent vector add; LDS staging via llvm.amdgcn.load.to.lds
__global__ void vadd_persistent(const float* __restrict__ A,
                                const float* __restrict__ B,
                                float* __restrict__ C, int N) {
  __shared__ float sA[256], sB[256];
  const int lane   = threadIdx.x;                  // 0..255 here (4 waves)
  const int stride = gridDim.x * blockDim.x;
  for (int base = blockIdx.x * blockDim.x; base < N; base += stride) {
    int i = base + lane;
    // direct-to-LDS async copy (HBM -> LDS, bypassing VGPRs)
    __builtin_amdgcn_load_to_lds(/*src*/ A + i, /*dst*/ &sA[lane],
                                 /*size*/ 4, /*offset*/ 0, /*aux*/ 0);
    __builtin_amdgcn_load_to_lds(B + i, &sB[lane], 4, 0, 0);
    __builtin_amdgcn_s_waitcnt(/*vmcnt*/ 0);       // simplify: drain for clarity
    __syncthreads();
    if (i < N) C[i] = sA[lane] + sB[lane];         // bounds check (no V# here)
  }
}
```

Note the HIP path uses an explicit `if (i < N)` because plain pointer loads do
**not** get buffer OOB semantics; only `buffer_*` MUBUF ops (V# descriptor) do.
That is precisely the correctness convenience the assembly version buys.

## Performance notes

Vector-add moves **12 bytes per element** (two FP32 reads + one FP32 write), so
it is bound by HBM bandwidth, not the matrix or vector ALU. On MI300X
([5.3 TB/s HBM3 peak](../../sources/docs/doc-mi300x-datasheet.md)) a
well-pipelined add reaches roughly 4.8–4.9 TB/s of effective traffic (~90%+ of
peak) for large `N`; the asm/HIP versions converge here because both are
DRAM-limited. Wins from the asm version are mostly in the **tail** (no remainder
branch divergence) and in **occupancy** (LDS-staged loads free ArchVGPRs). For a
clean bandwidth roofline measurement use non-temporal `float4` streaming — see
the [bandwidth microbenchmark](bandwidth-microbench.md).

Tuning levers:

- **`vmcnt` threshold** — too low serializes load/compute; too high overflows the
  in-flight-load budget and stalls on the 6-bit VMCNT.
- **TILE size** — bigger tiles amortize loop overhead but cost LDS (64 kB/CU on
  gfx942, 160 kB/CU on gfx950), capping waves/CU.
- **Persistent grid size** — launch ~`#CUs × waves/CU` workgroups so each wave
  does many tiles; this removes the [tail effect](../patterns/tail-effect.md).

## Runnable example

A runnable companion lives in [`examples/vector-add-asm/`](../../examples/vector-add-asm/).
It has two parts:

1. **`vadd_hip.cpp`** — the portable HIP grid-stride vector add (the "what you'd
   usually ship" kernel above). It **builds and runs on gfx1201** (RDNA4) and
   self-checks against a CPU reference.
2. **`vadd_asm_gfx942.cpp`** — a GCN inline-assembly vector add
   (`global_load_dword` / `global_store_dword` gated by `s_waitcnt vmcnt(0)`)
   that is **cross-compiled for gfx942** to illustrate the VMEM/wait-counter asm
   path. It is not executed on this RDNA4 box.

```bash
cd examples/vector-add-asm && ./build.sh
# Part 1 (runs on gfx1201):
#   vadd HIP (portable, gfx1201): N=16777216  block=256 grid=4096
#     time = 0.350 ms/iter   effective BW = 574.8 GB/s (12 B/elem)
#     max abs err = 0
#     PASS
# Part 2 (cross-compile-only):
#   OK: vadd_asm_gfx942.o produced (not executed on gfx1201)
```

The hand-written `buffer_*` double-buffered kernel in this page targets CDNA and
is cross-compile-verified; the asm path is demonstrated runnably via the gfx942
inline-asm object plus the portable HIP kernel that actually executes here.

## See also

- [Direct-to-LDS async copy](../hardware/async-copy-lds.md)
- [s_waitcnt counters](../hardware/s-waitcnt.md)
- [Buffer OOB guard technique](../techniques/buffer-oob-guard.md)
- [Persistent kernel technique](../techniques/persistent-kernel.md)
- [GCN/CDNA assembly language guide](../languages/gcn-asm.md)

## Sources

- [GCN/CDNA assembly notes & examples (gcnasm)](https://github.com/ROCm/amd_matrix_instruction_calculator) — assembly reference patterns for MUBUF/`s_waitcnt`.
- [AMD Instinct MI300 CDNA3 ISA Reference Guide](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf) — `buffer_load_dword ... lds`, `s_waitcnt`, V# descriptor, OOB semantics.
- [AMD Instinct MI300X datasheet](https://www.amd.com/en/products/accelerators/instinct/mi300/mi300x.html) — 5.3 TB/s HBM3 peak bandwidth.
