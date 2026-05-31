---
id: technique-vectorized-loads
title: Vectorized & Non-Temporal Loads (128-bit) to Saturate HBM
type: technique
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
adds, the A/B operands of a single-pass GEMM tail), add the **non-temporal**
hint so the load bypasses L2 residency policy and does not evict useful cache
lines. The combination — wide + non-temporal + enough outstanding loads to hide
latency — is what gets you close to the **5.3 TB/s** HBM3 ceiling on MI300X (and
up to **8 TB/s** HBM3E on MI355X).

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

// 16-byte aligned, 4 dwords -> compiles to global_load_dwordx4
using float4 = __attribute__((__vector_size__(16))) float;

// Streaming copy: each thread moves 16 B per iteration, grid-strided.
__global__ void copy_vec4(const float4* __restrict__ in,
                          float4* __restrict__ out,
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

For read-once data, use the LLVM AMDGPU non-temporal builtins so the access is
tagged streaming (sets the `slc`/non-temporal bits) and avoids polluting L2:

```cpp
// Read-once residual add: a[] is streamed, never revisited.
__global__ void residual_add_nt(const float4* __restrict__ a,
                                const float4* __restrict__ b,
                                float4* __restrict__ c, size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float4 va = __builtin_nontemporal_load(&a[i]);   // streaming 128-bit load
    float4 vb = __builtin_nontemporal_load(&b[i]);
    float4 vc;
    for (int k = 0; k < 4; ++k) vc[k] = va[k] + vb[k];
    __builtin_nontemporal_store(vc, &c[i]);          // streaming 128-bit store
}
```

`__builtin_nontemporal_load/store` are documented in the
[LLVM AMDGPU backend](../../sources/docs/doc-llvm-amdgpu.md). They are a *hint*:
they change cache replacement policy, not correctness. Use them only for data
with no temporal reuse — applying them to a tile that you re-read from L2 (e.g. a
GEMM operand reused across the K-loop) will *cost* you bandwidth.

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
