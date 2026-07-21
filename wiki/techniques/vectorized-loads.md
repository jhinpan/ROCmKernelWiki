---
id: technique-vectorized-loads
title: Vectorized & Non-Temporal Loads (128-bit) to Saturate HBM
type: technique
version_sensitive:
- vs-amdgpu-nontemporal-lowering
architectures:
- gfx942
- gfx950
- gfx1201
tags:
- vectorized-loads
- nontemporal-loads
- buffer-instructions
- global-instructions
- hbm3
- data-reuse
- memory-bound
confidence: source-reported
reproducibility: runnable
hardware_features:
- buffer-instructions
- global-instructions
- s-waitcnt
languages:
- hip
- gcn-asm
related:
- hw-memory-instructions
- kernel-bandwidth-microbench
- hw-s-waitcnt
- hw-async-copy-lds
- technique-buffer-oob-guard
- pattern-memory-bound
sources:
- ref-gcnasm
- doc-cdna3-isa
- doc-mi300x-datasheet
- blog-gemm-optimization
- doc-llvm-amdgpu
- blog-amdgpu-kernel-opt-guide
implemented_by:
- pr-aiter-2394
- pr-triton-729
- pr-composable_kernel-1430
- pr-aiter-3072
- pr-Tensile-293
- pr-Tensile-1521
- pr-Tensile-1288
- pr-Tensile-1185
---
# Vectorized & Non-Temporal Loads (128-bit) to Saturate HBM

## Overview

A memory-bound kernel on CDNA lives or dies by **how many bytes each VMEM
instruction moves** and **how many such instructions are in flight**. The single
most reliable lever is to issue **128-bit (16-byte) loads** — `dwordx4` /
`float4` — instead of scalar 32-bit loads. Each `global_load_dwordx4` or
`buffer_load_dwordx4` moves the same 16 bytes/lane that four separate 32-bit
loads would, but costs **one** instruction issue slot, one address calculation,
and one entry in the `VMCNT` queue. On a wave64 that is `64 lanes × 16 B = 1 KiB`
per instruction — the natural granularity for streaming HBM.

For data that is read **once and never reused** (streaming copies, residual
adds, the A/B operands of a single-pass GEMM tail), a **non-temporal** hint can
make it less likely that the stream displaces useful cache lines. The exact
AMDGPU cache-policy encoding and which level is affected are target/compiler
specific; non-temporal is not a promise to bypass every cache and does not alter
coherency semantics. Wide accesses plus enough outstanding work are what get a
kernel close to the **5.3 TB/s** HBM3 ceiling on MI300X (and up to **8 TB/s**
HBM3E on MI355X).

## Why width matters

The HBM-facing path issues one VMEM instruction per cycle per SIMD at best.
Latency to HBM is hundreds of cycles, so throughput is bounded by
`bytes_per_instruction × instructions_in_flight / latency`. Widening the access
multiplies the numerator for free:

| Load form | Bytes/lane | Bytes/wave64 | VMCNT entries for 16 B/lane |
|---|---|---|---|
| `global_load_dword`    | 4  | 256 B  | 4 |
| `global_load_dwordx2`  | 8  | 512 B  | 2 |
| `global_load_dwordx4`  | 16 | 1 KiB  | 1 |

Fewer instructions for the same bytes means less issue pressure, fewer address
ALU ops, and a shorter `VMCNT` queue (6-bit, so at most 63 outstanding) spent on
useful width rather than count. See [s_waitcnt](../hardware/s-waitcnt.md) for how
`VMCNT` gates these loads.

## HIP: 128-bit vectorized loads

The compiler emits a `*_dwordx4` whenever it can prove the access is **16-byte
aligned** and contiguous. The most robust way to force it is to load through a
16-byte vector type:

```cpp
#include <hip/hip_runtime.h>

// A Clang-native vector (unlike HIP's float4 wrapper) is accepted by the
// nontemporal builtins and compiles to one global_load/store_dwordx4.
using f32x4 = float __attribute__((ext_vector_type(4)));

// Streaming copy: each thread moves 16 B per iteration, grid-strided.
__global__ void copy_vec4(const f32x4* __restrict__ in,
                          f32x4* __restrict__ out,
                          size_t n_vec4 /* = n_floats / 4 */) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = size_t(gridDim.x) * blockDim.x;
    for (; i < n_vec4; i += stride) {
        out[i] = in[i];          // load_dwordx4 ; store_dwordx4
    }
}
```

Two correctness preconditions for the wide form to actually be emitted:

* **Alignment.** The pointer must be 16-byte aligned. `hipMalloc` returns at
  least 256-byte alignment, but offsets into a buffer must keep the 16-byte
  multiple. If you index by `float`, the compiler conservatively falls back to
  `dwordx1`.
* **No aliasing.** Mark inputs `const __restrict__` so the compiler may keep
  multiple loads in flight without inserting `VMCNT` waits between them.

## HIP: non-temporal (streaming) loads

For read-once data, use the LLVM non-temporal builtins so the access is tagged
as streaming. Inspect the target ISA to see how the current backend encoded the
hint; do not assume a particular `slc`/`nt` bit or cache-bypass behavior across
gfx generations:

```cpp
// Read-once residual add: a[] is streamed, never revisited.
__global__ void residual_add_nt(const f32x4* __restrict__ a,
                                const f32x4* __restrict__ b,
                                f32x4* __restrict__ c, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    f32x4 va = __builtin_nontemporal_load(&a[i]);   // streaming 128-bit load
    f32x4 vb = __builtin_nontemporal_load(&b[i]);
    f32x4 vc;
    for (int k = 0; k < 4; ++k) vc[k] = va[k] + vb[k];
    __builtin_nontemporal_store(vc, &c[i]);          // streaming 128-bit store
}
```

`__builtin_nontemporal_load/store` are documented in the
[LLVM AMDGPU backend](../../sources/docs/doc-llvm-amdgpu.md). They are a *hint*:
they request non-temporal treatment without changing correctness. Use them only
for data with no temporal reuse — applying them to a tile that you intend to
re-read from cache (for example a GEMM operand reused across the K-loop) can
cost bandwidth.

> **Compiler check (2026-07-20).** ROCm 7.1.1 / clang 20 compiled the native
> `f32x4` form above for both gfx942 and gfx950 as
> `global_load_dwordx4 ... nt` / `global_store_dwordx4 ... nt`. HIP's built-in
> `float4` wrapper was rejected by these Clang non-temporal builtins, which is
> why the snippet deliberately uses `ext_vector_type(4)`. Re-check final ISA
> when changing compiler or target.

## Assembly: what to look for

The whole technique is verifiable in one `rocprof`/`llvm-objdump` pass — the hot
loop should contain wide loads, ideally clustered before a single `s_waitcnt`:

```asm
; streaming copy inner loop (CDNA, simplified) — see ref-gcnasm
  global_load_dwordx4  v[8:11],  v[0:1], off       ; 16 B/lane in flight
  global_load_dwordx4  v[12:15], v[2:3], off
  s_waitcnt            vmcnt(0)                      ; gate on both loads
  global_store_dwordx4 v[4:5],   v[8:11], off
```

If you instead see a run of `global_load_dword` (x1) with a `s_waitcnt vmcnt(0)`
after *every* load, the vectorization or alignment precondition failed — the
kernel is leaving >50% of HBM bandwidth on the table. The same pattern applies to
`buffer_load_dwordx4` when addressing through a V# resource descriptor (which
additionally gives free [out-of-bounds guards](buffer-oob-guard.md); see
[memory instructions](../hardware/memory-instructions.md)).

## Tuning checklist

1. **Widen first.** Cast to a 16-byte vector type and confirm `*_dwordx4` in the
   ISA dump before tuning anything else.
2. **Keep loads in flight.** Issue several wide loads, then a single
   `s_waitcnt vmcnt(N)` — do not wait after each load. Unroll the grid-stride
   loop 2–4× so the `VMCNT` queue stays populated.
3. **Use grid-stride loops**, sizing the grid to a small multiple of the CU
   count so every CU streams continuously (avoids the tail effect).
4. **Apply non-temporal only to read-once data.** Reused tiles belong in L2/LDS;
   stream the rest.
5. **Mind alignment at the boundary.** Handle the ragged tail (`n % 4`) with a
   scalar epilogue or a [buffer OOB guard](buffer-oob-guard.md) rather than
   de-vectorizing the whole loop.
6. **Measure against the roofline.** Compare achieved GB/s to the
   [bandwidth microbenchmark](../kernels/bandwidth-microbench.md) — if you are
   below ~80% of its number, you are instruction-bound, not bandwidth-bound.

## Performance notes

A pure copy/triad written with 128-bit non-temporal loads and a grid-stride loop
reaches a large fraction of the **5.3 TB/s** MI300X HBM3 peak reported in the
[datasheet](../../sources/docs/doc-mi300x-datasheet.md); the scalar `dword`
version typically tops out near half of that because it saturates instruction
issue before it saturates memory. The same code recompiled for gfx950 scales
toward the **8 TB/s** HBM3E ceiling. These are bandwidth-bound observations
(`confidence: source-reported`) — exact percentages depend on grid sizing,
unroll factor, and L2/Infinity-Cache hit rate. See the
[GEMM optimization blog](../../sources/blogs/blog-gemm-optimization.md) for how
wide A/B loads feed the LDS double-buffer stage in a real kernel.

## Coalescing checklist (from the shark-ai optimization guide)

The nod-ai/shark-ai *AMDGPU Kernel Optimization Guide* gives a precise recipe for
saturating the memory path on GFX9. Treat it as a checklist when a kernel is
memory-bound (see [memory-bound pattern](../patterns/memory-bound.md)):

1. **Use 16 B / 128-bit accesses** — `global_load_dwordx4` / `global_store_dwordx4`.
2. **Make the access subgroup-contiguous** so the whole 64-lane wave touches
   a **1 KiB payload** per `dwordx4` (`64 lanes × 16 B`). The guide's 512 B
   figure corresponds to wave64×8 B or wave32×16 B, not wave64 dwordx4.
3. **Keep candidate clauses adjacent:** the guide reports that up to four
   neighboring `global_load_dwordx4` instructions can be treated as one clause
   and reduce data-fabric transactions. This is **guide-reported and
   target/compiler-sensitive**; confirm final ISA and fabric counters rather
   than assuming one transaction.
4. **Cover all four L1D sets:** the guide recommends four distinct 128 B lines
   per workgroup. Treat the mapping as a layout experiment and verify with cache
   counters.
5. **Launch enough work:** make the grid a **multiple of the CU count** and engage
   all XCDs and memory interfaces. MI300X has **4 IODs**; MI355X has **2**, so
   do not hardcode four as a cross-generation rule.
6. **Use non-temporal** loads/stores for streamed, write-once / read-once data so
   the backend can apply an appropriate cache-policy hint. It does not disable
   coherency or guarantee bypass of every cache.

These rules are exactly what the [bandwidth microbenchmark](../kernels/bandwidth-microbench.md)
exploits to reach multi-TB/s on MI300-class parts.

## See also

- [Memory instructions: buffer vs global vs flat](../hardware/memory-instructions.md)
- [Bandwidth microbenchmark (float4 non-temporal)](../kernels/bandwidth-microbench.md)
- [s_waitcnt async gating](../hardware/s-waitcnt.md)
- [Direct-to-LDS async copy](../hardware/async-copy-lds.md)
- [Memory-bound pattern](../patterns/memory-bound.md)

## Sources

- [AMD GCN/CDNA Assembly Reference (gcnasm)](https://gpuopen.com/learn/amd-gcn-assembly-and-the-cdna-architecture/)
- [CDNA3 ISA Reference Guide](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf)
- [AMD Instinct MI300X Datasheet](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/data-sheets/amd-instinct-mi300x-data-sheet.pdf)
- [LLVM User Guide for AMDGPU Backend](https://llvm.org/docs/AMDGPUUsage.html)
- [Optimizing GEMM on AMD GPUs](https://rocm.blogs.amd.com/artificial-intelligence/matrix-cores/README.html)
- [AMDGPU Kernel Optimization Guide (captured snapshot)](https://github.com/nod-ai/amd-shark-ai/blob/efa471aeef66a260c85983cc41e833bfa769dade/docs/amdgpu_kernel_optimization_guide.md) — coalescing, cache-set, and clause recommendations (with the qualifications above)
