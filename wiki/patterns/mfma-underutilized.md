---
id: pattern-mfma-underutilized
title: Matrix Cores Idle — MFMA Underutilization
type: pattern
architectures:
- gfx942
- gfx950
tags:
- mfma
- matrix-core
- compute-bound
- mfma-pipelining
- lds-double-buffering
- agpr
- co-issue
symptoms:
- mfma-underutilized
- compute-bound
- matrix-idle
candidate_techniques:
- technique-mfma-pipelining
- technique-lds-double-buffering
- technique-preshuffle-layout
related:
- hw-mfma
- hw-lds
- technique-mfma-pipelining
- technique-lds-double-buffering
- technique-preshuffle-layout
- kernel-ck-hgemm
- pattern-vgpr-pressure
sources:
- hw-mfma
- blog-4wave-fp8-gemm
- blog-gemm-optimization
- blog-matrix-cores-cdna
- doc-mi300x-datasheet
implemented_by:
- pr-composable_kernel-2110
- pr-Tensile-1288
---
# Matrix Cores Idle — MFMA Underutilization

## Symptom

A GEMM-shaped or attention-shaped kernel is *compute-bound on paper* — the
arithmetic intensity is high, the problem is large enough to fill all CUs, and
HBM bandwidth is **not** the limiter — yet achieved TFLOPS sit far below the
[MI300X / MI355X matrix-core peaks](../../sources/docs/doc-mi300x-datasheet.md).
The matrix unit (`v_mfma_*`) is *issuing*, but it is **starved**: it spends most
cycles waiting on operands instead of running back-to-back.

Tell-tale signs:

- `rocprof` / `rocprofv3` shows high `VALUBusy` or high `LDSBusy`/`MemUnitStalled`
  but a low ratio of MFMA-issue cycles to total active cycles.
- The inner loop body shows `v_mfma_*` instructions separated by long
  `s_waitcnt lgkmcnt(0)` / `s_waitcnt vmcnt(0)` stalls (the wave is blocking on
  LDS reads or global loads feeding the *next* MFMA).
- Increasing the tile's K-depth or batch does **not** improve TFLOPS — you are
  not memory-bound, you are *latency-bound on the operand feed*.
- A roofline places the kernel under the compute roof but well left of it.

This is distinct from [`pattern-memory-bound`](memory-bound.md) (HBM is the wall)
and [`pattern-vgpr-pressure`](vgpr-pressure.md) (occupancy collapse from register
spills). Here the bottleneck is the *scheduling of operand delivery into the
matrix core*, not raw bandwidth or occupancy.

## Why it happens

An MFMA executes across an entire **64-lane wavefront**, consuming A/B fragments
out of VGPRs and accumulating into AGPRs (see [MFMA](../hardware/mfma.md)). For
`v_mfma_f32_16x16x16_f16` the unit retires 8192 FLOPs in ~16 cycles
(2048 FLOPs/CU/cycle); `v_mfma_f32_16x16x32_fp8_fp8` doubles that to
16384 FLOPs (4096 FLOPs/CU/cycle). To stay near peak, the operand pipeline must
deliver the *next* tile of A and B before the *current* MFMA group finishes.

The matrix core goes idle when that pipeline serializes:

1. **No global→compute overlap.** A naive loop does
   `load(K-tile) → s_waitcnt → mfma → load(next) → ...`. The MFMAs cannot start
   until the loads land, and the loads do not start until the MFMAs of the
   previous iteration are consumed. CDNA has no `mbarrier`/TMA; overlap is the
   programmer's job via [`s_waitcnt`](../hardware/s-waitcnt.md) counters.
2. **No LDS double buffering.** With a single LDS staging buffer, the
   `__syncthreads()` between *writing the next tile* and *reading the current
   tile* forces all waves to drain the matrix pipe before the next LDS fill.
3. **Bad LDS read layout.** Bank conflicts on `ds_read_b128` (see
   [LDS](../hardware/lds.md)) stretch the operand fetch, inserting bubbles
   between MFMAs even when the data is already on-chip.
4. **Accumulator too small / too large.** A tiny accumulator tile gives the
   scheduler nothing to hide load latency behind; an oversized one spills AGPRs
   and crushes occupancy ([vgpr-pressure](vgpr-pressure.md)).

## How to confirm

Inspect the compiled inner loop and the profiler counters:

```bash
# 1. Dump ISA and look for mfma grouped tightly vs. broken up by waits
hipcc -O3 --offload-arch=gfx942 -S -o gemm.s gemm.cpp
# Healthy: runs of v_mfma_* with few intervening s_waitcnt
# Sick:    every v_mfma_* preceded by s_waitcnt lgkmcnt(0) / vmcnt(0)

# 2. Profile: matrix-issue cycles vs. stall cycles
rocprofv3 --pmc VALUBusy MfmaUtil LDSBusy MemUnitStalled SQ_WAIT_INST_LDS \
    -- ./gemm
```

A kernel with the disease shows low MFMA utilization alongside high
`SQ_WAIT_INST_LDS` (waves blocked on LDS) or high `MemUnitStalled` — the matrix
core is idle behind the operand feed, not behind arithmetic.

## Candidate fixes

Apply in roughly this order; each targets a different bubble source.

### 1. Pipeline the MFMA / load schedule → [`technique-mfma-pipelining`](../techniques/mfma-pipelining.md)

Software-pipeline so loads for iteration *k+1* are in flight while the matrix
core chews iteration *k*. Use relaxed `s_waitcnt vmcnt(N)` (wait for all-but-N
outstanding loads) instead of `vmcnt(0)`, and hint the scheduler:

```cpp
// Prefetch next K-tile into registers/LDS, then issue MFMAs on the current tile.
// s_waitcnt vmcnt(N) keeps N loads in flight to overlap with the matrix pipe.
for (int k = 0; k < K_tiles; ++k) {
    prefetch_global_to_lds(a_next, b_next, k + 1);   // direct-to-LDS, no VGPR round-trip
    __builtin_amdgcn_s_waitcnt(/*lgkmcnt*/ 0 /*current LDS tile ready*/);
    #pragma unroll
    for (int kk = 0; kk < KK; ++kk)
        acc = __builtin_amdgcn_mfma_f32_16x16x16f16(a[kk], b[kk], acc, 0, 0, 0);
    __builtin_amdgcn_sched_barrier(0);   // keep mfma cluster intact across the swap
    swap(cur, next);
}
```

On gfx942 this is exactly the "ping-pong" schedule that the Triton AMD backend
and CK both emit; the [4-wave FP8 GEMM walkthrough](../../sources/blogs/blog-4wave-fp8-gemm.md)
shows interleaving global loads, LDS writes, and MFMAs across waves so the matrix
unit never drains.

### 2. Double-buffer LDS → [`technique-lds-double-buffering`](../techniques/lds-double-buffering.md)

Allocate two LDS tiles and alternate: write tile *k+1* while reading tile *k*.
This removes the global-store→`__syncthreads`→MFMA serialization. gfx942 has
64 kB LDS/CU and gfx950 has 160 kB — the larger budget on CDNA4 lets you keep
deeper buffers (or wider tiles) resident, which directly raises the steady-state
MFMA-issue ratio. Pair with [direct-to-LDS async copy](../hardware/async-copy-lds.md)
to fill the back buffer without burning VGPRs.

### 3. Pre-shuffle operands → [`technique-preshuffle-layout`](../techniques/preshuffle-layout.md)

Even with perfect pipelining, conflicted `ds_read` patterns insert bubbles
*between* MFMAs. Re-pack A/B in HBM (or during the LDS write) into the exact
lane/register order the MFMA expects, so the inner loop uses conflict-free
`ds_read_b128` and feeds the matrix core every cycle. Derive the target layout
with the AMD Matrix Instruction Calculator (see [MFMA](../hardware/mfma.md)).

## Expected payoff

The [GEMM optimization blog](../../sources/blogs/blog-gemm-optimization.md) walks
a HIP GEMM from a naive operand-starved loop to a pipelined, double-buffered,
layout-aware kernel and recovers the large majority of the matrix-core roofline
on MI300X. As a rule of thumb: pipelining + double buffering removes the
load-induced bubbles, and pre-shuffling removes the LDS-read bubbles; together
they move a "compute-bound but slow" kernel from a small fraction of peak toward
the [datasheet TFLOPS](../../sources/docs/doc-mi300x-datasheet.md). Verify with
the same `rocprofv3` counters — MFMA utilization should rise and
`SQ_WAIT_INST_LDS` should fall.

## See also

- [MFMA — Matrix Core Instructions](../hardware/mfma.md)
- [Local Data Share](../hardware/lds.md) and [s_waitcnt](../hardware/s-waitcnt.md)
- [VGPR/AGPR pressure pattern](vgpr-pressure.md) — the failure mode when you over-tile to feed the matrix core
- [CK FP16 GEMM kernel](../kernels/ck-hgemm.md)

## Sources

- [MFMA — AMD Matrix Core Instructions (CDNA)](../hardware/mfma.md)
- [A 4-Wave FP8 GEMM Schedule on CDNA3](https://rocm.blogs.amd.com/artificial-intelligence/fp8-gemm/README.html)
- [Optimizing GEMM on AMD GPUs](https://rocm.blogs.amd.com/artificial-intelligence/gemm-optimization/README.html)
- [Matrix Core Programming on CDNA3 and CDNA4](https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores-cdna/README.html)
- [AMD Instinct MI300X Datasheet](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/data-sheets/amd-instinct-mi300x-data-sheet.pdf)
