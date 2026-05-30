---
id: technique-mfma-pipelining
title: "MFMA Software Pipelining (interleaving loads and matrix issue)"
type: technique
architectures:
- gfx942
- gfx950
tags:
- mfma-pipelining
- software-pipelining
- mfma
- agpr
- s-waitcnt
- co-issue
- async-pipeline
- gemm
confidence: source-reported
reproducibility: snippet
hardware_features:
- mfma
- agpr
- s-waitcnt
- async-copy
- lds
kernel_types:
- gemm
- hgemm
- fp8-gemm
languages:
- hip
- gcn-asm
related:
- hw-mfma
- hw-s-waitcnt
- technique-lds-double-buffering
- technique-vgpr-budgeting
- pattern-mfma-underutilized
sources:
- hw-mfma
- hw-s-waitcnt
- blog-4wave-fp8-gemm
- blog-gemm-optimization
- doc-cdna3-isa
---

# MFMA Software Pipelining

## The problem

A single `v_mfma_*` keeps the matrix unit busy for a fixed number of cycles
(e.g. `v_mfma_f32_16x16x16_f16` runs for **16 cycles**, see
[MFMA](../hardware/mfma.md)). If the wavefront stalls on a `s_waitcnt` waiting
for the next K-slice of A/B to arrive from LDS or HBM *before* it can issue the
next MFMA, the matrix core sits idle and the kernel becomes
[MFMA-underutilized](../patterns/mfma-underutilized.md). On a GEMM main loop the
goal is the opposite: the matrix unit should issue back-to-back MFMAs while the
memory subsystem streams the *next* tile in the background.

AMD CDNA has **no `mbarrier`, no TMA, and no hardware warp scheduler reordering
across long latencies**. Overlap is created by the compiler's instruction
schedule plus the in-order async counters (`vmcnt`/`lgkmcnt`, see
[s_waitcnt](../hardware/s-waitcnt.md)). Software pipelining means *manually
arranging* the load → wait → MFMA sequence — and, where the compiler scheduler
gets it wrong, *fencing* it with `__builtin_amdgcn_sched_barrier`.

## Pattern 1 — prefetch + relaxed waitcnt

Issue all loads for K-slice `k+1` *before* consuming K-slice `k`, then wait only
for the slice you are about to multiply. Because same-type VMEM/LDS ops retire
**in order**, `s_waitcnt lgkmcnt(N)` lets the oldest reads drain while newer
prefetches stay in flight.

```cpp
// FP16 GEMM inner loop, double-buffered LDS, 1 MFMA shape = 16x16x16.
// a_frag/b_frag: [2] ping-pong register buffers filled from LDS.
// acc: float4 accumulator held in AGPRs across the whole K loop.
#pragma clang loop unroll(disable)
for (int k = 0; k < K_TILES; ++k) {
    int cur = k & 1, nxt = (k + 1) & 1;

    // Kick off the LDS reads for the NEXT k-slice (non-blocking).
    if (k + 1 < K_TILES) {
        a_frag[nxt] = ds_read_b64(lds_a + off_a(k + 1));
        b_frag[nxt] = ds_read_b64(lds_b + off_b(k + 1));
    }

    // Wait only until everything except the 2 freshly-issued reads has landed.
    __builtin_amdgcn_s_waitcnt(/*lgkmcnt=*/0x0000 | 2);

    // Consume the CURRENT slice while the next is still streaming.
    acc = __builtin_amdgcn_mfma_f32_16x16x16f16(a_frag[cur], b_frag[cur], acc,
                                                /*cbsz=*/0, /*abid=*/0, /*blgp=*/0);
}
```

In practice you let the compiler manage the literal `lgkmcnt` immediate (encode
it via the `__builtin_amdgcn_s_waitcnt` form or simply rely on the backend);
the structural point is **prefetch-ahead-by-one** so a load latency is hidden
behind a full MFMA.

## Pattern 2 — `sched_barrier` to lock the schedule

The LLVM AMDGPU scheduler will happily hoist or sink instructions across your
carefully placed loads, sometimes collapsing the overlap you built. A
zero-overhead `__builtin_amdgcn_sched_barrier(mask)` creates a scheduling fence:
the compiler may not move instructions of the masked classes across it. `mask=0`
forbids *all* reordering across the barrier; bit flags allow selected classes
(e.g. keep VALU free to move but pin MFMA and memory).

```cpp
// Force: all NEXT-tile global loads issue, THEN a block of MFMAs, with no
// reordering between the two groups.
global_load_dwordx4(reg_next, gptr + tile_stride);   // direct-to-VGPR prefetch
__builtin_amdgcn_sched_barrier(0);                   // hard fence: nothing crosses

#pragma unroll
for (int i = 0; i < MFMA_PER_TILE; ++i)
    acc[i] = __builtin_amdgcn_mfma_f32_16x16x16f16(a[i], b[i], acc[i], 0, 0, 0);

__builtin_amdgcn_sched_barrier(0);
__builtin_amdgcn_s_waitcnt(0);                        // now consume prefetched regs
```

A companion intrinsic, `__builtin_amdgcn_sched_group_barrier(mask, size, sync_id)`,
lets you describe an explicit *issue pattern* — "N MFMAs, then M `ds_read`s, then
repeat" — which is how Triton's AMD backend and Composable Kernel emit their
interleaved GEMM cores.

## Pattern 3 — 4-wave interleave (hiding latency with occupancy)

A single wave cannot always cover MFMA + memory latency from its own
instruction stream. The complementary lever is **occupancy**: schedule multiple
waves per SIMD so that while wave 0 waits on `vmcnt`, the SIMD issues MFMAs from
waves 1–3. AMD's FP8 GEMM work on CDNA3 uses a **4-wave interleave** in which
four wavefronts cooperatively tile the output and round-robin their MFMA issue,
keeping the matrix unit saturated even though each individual wave periodically
stalls. With 4 SIMDs per CU this maps to one busy wave per SIMD per cycle.

Practical levers that create this on CDNA3/CDNA4:

- Size the accumulator tile so **≥4 waves/SIMD** survive the AGPR + LDS budget
  (each `v_mfma_f32_16x16x16_f16` accumulator is 4 AGPRs/lane; see
  [VGPR/AGPR budgeting](vgpr-budgeting.md)).
- Pair with [LDS double-buffering](lds-double-buffering.md) so each wave always
  has a ready tile.
- On gfx950, prefer [direct-to-LDS async copy](../hardware/async-copy-lds.md)
  for the HBM→LDS leg, freeing VGPRs and decoupling the streaming engine from
  the matrix issue.

## Co-issue: why this works on CDNA

The matrix core and the VALU/memory pipes are separate issue ports. While an
MFMA occupies the matrix unit for its 16 cycles, the same wave (or a sibling
wave) can issue `ds_read`/`buffer_load`/address-VALU in the shadow. Software
pipelining exists to *fill that shadow*. The accumulator must stay resident in
AGPRs for the whole K loop — spilling it defeats the pipeline, so accumulator
size is the primary trade against the occupancy that Pattern 3 needs.

## Checklist

1. Prefetch the next K-slice (LDS and/or HBM) before issuing the current MFMA.
2. Relax `s_waitcnt` to `cnt(N>0)` so prefetches stay in flight.
3. Keep the full accumulator in AGPRs across the loop — never spill it.
4. If the compiler reshuffles your overlap, pin it with
   `__builtin_amdgcn_sched_barrier` / `sched_group_barrier`.
5. Provision ≥4 waves/SIMD so independent waves cover residual stalls.
6. Inspect `rocprofv3` / `--detail-instruction` and the generated assembly to
   confirm MFMAs issue back-to-back with no idle bubbles.

## See also

- [MFMA matrix instructions](../hardware/mfma.md)
- [s_waitcnt async gating](../hardware/s-waitcnt.md)
- [LDS double-buffering](lds-double-buffering.md)
- [MFMA-underutilized pattern](../patterns/mfma-underutilized.md)

## Sources

- [CDNA3 ISA Reference Guide](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf)
- [Optimizing GEMM kernels on AMD GPUs](https://rocm.blogs.amd.com/software-tools-optimization/gemm-optimization/README.html)
- [A 4-wave-interleaved FP8 GEMM on CDNA3](https://rocm.blogs.amd.com/artificial-intelligence/fp8-gemm/README.html)
- [LLVM AMDGPU `sched_barrier` / `s_waitcnt` semantics](https://llvm.org/docs/AMDGPUUsage.html)
