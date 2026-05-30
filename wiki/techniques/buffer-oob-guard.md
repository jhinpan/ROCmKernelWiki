---
id: technique-buffer-oob-guard
title: "Branchless Boundary Handling with Buffer OOB Guards"
type: technique
architectures:
- gfx942
- gfx950
- gfx1201
tags:
- buffer-oob-guard
- buffer-instructions
- vectorized-loads
- gcn-asm
- cdna
- predication
confidence: source-reported
reproducibility: snippet
hardware_features:
- buffer-instructions
- global-instructions
languages:
- hip
- gcn-asm
related:
- hw-memory-instructions
- technique-vectorized-loads
- hw-s-waitcnt
- lang-gcn-asm
- lang-triton-amd
sources:
- doc-cdna3-isa
- hw-memory-instructions
- ref-gcnasm
- doc-llvm-amdgpu
- blog-triton-amd
---

# Branchless Boundary Handling with Buffer OOB Guards

## The problem: tail tiles and ragged bounds

Almost every tiled kernel has a *boundary* problem. A GEMM with `M=4097` and a
tile of 256 rows has a final tile that is only 1 row tall; a reduction over an
arbitrary-length vector has a ragged last chunk. The textbook fix is a
per-element predicate:

```cpp
if (row < M && col < N)
    acc += A[row * lda + col];   // scalar branch per work-item
```

On a 64-lane wavefront this is corrosive. A divergent `if` forces the lanes that
fail the test to mask off, and (worse) the address computation `A + row*lda+col`
is still evaluated on those lanes — an out-of-bounds address that you must *never*
dereference. Programmers paper over this with `min()`/clamping or duplicate
"main loop + tail loop" code paths, both of which add VALU work and instruction
cache pressure to the steady state.

CDNA/RDNA give you a hardware alternative: let the **memory system** do the bounds
check for free.

## Buffer instructions and `num_records`

`buffer_load`/`buffer_store` (the MUBUF class) address memory through a 128-bit
**resource descriptor** (a "V#") that carries a `base_address`, a `stride`, and a
`num_records` field. The hardware computes the effective byte offset and compares
it against the extent implied by `num_records × stride`. The ISA guarantees:

> An out-of-bounds **read returns 0** (or `1.0` for components selected by
> `dst_sel = SEL_1`); an out-of-bounds **write is dropped**.

That is the whole trick. If you *size the buffer descriptor to the real tensor
extent*, then any lane whose index runs past the end reads a clean `0` and any
overhanging store silently evaporates — no branch, no mask, no clamp, and no
illegal address fault. This is exactly why the AMD compiler's "buffer ops" passes
prefer MUBUF for in-bounds-uncertain accesses. Contrast with
[`global_load`/`flat`](../hardware/memory-instructions.md), which take a raw
64-bit pointer and have **no** `num_records` guard — an OOB flat access is a real
fault (`MEM_VIOL`), so those paths still need software predication.

## Building the descriptor in HIP

You rarely hand-assemble a V#. The portable way to get OOB-guarded loads in HIP
is the `__builtin_amdgcn_make_buffer_rsrc` + `__builtin_amdgcn_raw_buffer_load`
intrinsics (a.k.a. the LLVM `amdgcn.raw.ptr.buffer.load` family). Size the
descriptor with the number of **bytes** of valid data:

```cpp
#include <hip/hip_runtime.h>

// Branchless guarded load of one float; OOB element reads back as 0.0f.
__device__ inline float guarded_load(const float* base,
                                     int elem_idx,      // may be >= n_elems
                                     int n_elems)
{
    // make_buffer_rsrc(ptr, stride/*=0 for raw*/, num_records_bytes, flags)
    void* rsrc = __builtin_amdgcn_make_buffer_rsrc(
        (void*)base,
        /*stride=*/   0,
        /*num_records=*/ n_elems * (int)sizeof(float),  // valid extent in BYTES
        /*flags=*/    0);

    // raw_buffer_load_f32(rsrc, voffset, soffset, aux)
    return __builtin_amdgcn_raw_buffer_load_f32(
        rsrc,
        /*voffset=*/ elem_idx * (int)sizeof(float),
        /*soffset=*/ 0,
        /*aux=*/     0);   // lane past num_records -> hardware returns 0.0f
}
```

Every lane runs the *same* code; lanes whose `elem_idx >= n_elems` simply receive
`0.0f`. No `if`, no `EXEC` divergence, no separate tail loop. For a row-major 2-D
tile, fold both dimensions into the byte offset and set `num_records` to the
full tensor byte size so that *both* the M and N overhang are guarded by the one
descriptor.

## What it looks like in assembly

The same idea, written directly in GCN/CDNA assembly (see
[`ref-gcnasm`](../../sources/refs/ref-gcnasm.md)), is a single `buffer_load_dword`
against an SGPR-resident descriptor `s[0:3]`:

```asm
; s[0:1] = base addr, s2 = num_records (bytes), s3 = format/flags word
; v0     = per-lane byte offset (elem_idx * 4), may exceed s2
    buffer_load_dword v1, v0, s[0:3], 0 offen   ; OOB lanes -> v1 = 0
    s_waitcnt vmcnt(0)                          ; wait for the load to land
    ; ... use v1; out-of-range lanes already hold 0, no masking needed
```

The descriptor's `num_records` word is the guard. Because the check is in the
memory pipe, the load issues for the full wavefront in one instruction — the
in-bounds and out-of-bounds lanes share the same VMEM transaction and the same
[`s_waitcnt vmcnt`](../hardware/s-waitcnt.md) retirement.

## When zero is the wrong fill

The OOB-returns-0 contract is ideal for **additive** accumulation (GEMM tiles,
dot products, softmax numerator) and for masked-MoE gathers, because `0`
contributes nothing to a sum and `x·0 = 0` to a matrix product. It is *not* a
free lunch when the neutral element is not zero:

- **`max`/`min` reductions** (e.g. the running max in FlashAttention) need
  `-inf`/`+inf`, not `0`. Pre-fill the accumulator with the correct identity, or
  guard the *index range* of the reduction loop so the OOB lanes never enter the
  `max`.
- **Stores into a larger destination** rely on OOB-write-dropping, which is
  always safe — but make sure padding bytes that *will* be read later were
  initialized, since a dropped write leaves stale data.
- **Division / reciprocal** of a guarded-loaded denominator can produce `inf`;
  add the bias before the divide.

## Interaction with vectorization and occupancy

Buffer OOB guards compose cleanly with
[vectorized loads](vectorized-loads.md): a `buffer_load_dwordx4` past the end
returns a fully-zeroed `float4`, so you keep 128-bit transactions right up to the
ragged edge instead of dropping to scalar tail code. Eliminating the tail loop
also shrinks the kernel's instruction footprint, which helps I-cache residency
and removes the per-iteration compare from the hot loop — a measurable win on
memory-bound elementwise and reduction kernels where the VALU compare was on the
critical path. The Triton AMD backend exposes this via its *buffer-ops* lowering
(`tl.load` with a `mask`/`other=0.0` lowers to guarded `buffer_load` when the
pointer is provably a tensor base plus offset — see
[`blog-triton-amd`](../languages/triton-amd.md)).

## Pitfalls

- **`num_records` is in the descriptor's units.** For a "raw" (`stride=0`) buffer
  it is a *byte* count; for a "structured" buffer it is a record count multiplied
  by `stride`. Mixing the two silently truncates the valid region.
- **It guards range, not alignment.** A misaligned `buffer_load_dwordx4` is still
  a fault; the OOB guard does not relax alignment rules.
- **Flat/global paths are unguarded.** If a pass demotes your `buffer_load` to a
  `global_load` (e.g. because the descriptor couldn't be proven loop-invariant),
  the OOB protection is gone. Inspect the assembly for `buffer_` vs `global_`.
- **Negative offsets wrap.** The comparison is unsigned; a negative `voffset`
  becomes a huge positive value (still OOB → 0), which is usually what you want,
  but do not rely on signed clamping.

## See also

- [Memory instructions: buffer vs global vs flat](../hardware/memory-instructions.md)
- [Vectorized loads](vectorized-loads.md)
- [s_waitcnt async gating](../hardware/s-waitcnt.md)
- [GCN/CDNA assembly](../languages/gcn-asm.md)

## Sources

- [AMD Instinct MI300/CDNA3 ISA Reference Guide — MUBUF / buffer resource descriptor, out-of-range behavior](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf)
- [LLVM AMDGPU backend — `llvm.amdgcn.raw.ptr.buffer.load` / `make.buffer.rsrc` intrinsics](https://llvm.org/docs/AMDGPUUsage.html)
- [gcnasm — worked buffer_load examples](https://github.com/AMD/gcnasm)
- [Triton AMD backend — buffer-ops lowering for masked loads](https://rocm.blogs.amd.com/artificial-intelligence/triton/README.html)
