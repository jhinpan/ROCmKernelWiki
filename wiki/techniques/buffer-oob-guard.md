---
id: technique-buffer-oob-guard
title: Branchless Boundary Handling with Buffer OOB Guards
type: technique
architectures:
- gfx942
- gfx950
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
- doc-cdna4-isa
- hw-memory-instructions
- ref-gcnasm
- doc-llvm-amdgpu
- blog-triton-amd
- blog-amdgpu-kernel-opt-guide
implemented_by:
- pr-triton-729
- pr-aiter-2328
- pr-aiter-2685
- pr-composable_kernel-3512
- pr-composable_kernel-2425
- pr-Tensile-1288
- pr-FlyDSL-197
- pr-FlyDSL-131
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
`num_records` field. The exact range comparison depends on the addressing mode:

- a **raw** buffer checks
  `InstOffset + (OffEN ? vgpr_offset : 0) >= NumRecords`, with `NumRecords` in
  bytes; and
- a **structured** buffer checks `Index(vgpr) >= NumRecords`, with
  `NumRecords` counting records. `stride` participates in address formation,
  not in that index comparison.

The SGPR `soffset` contributes to the final address but is not part of the raw
range comparison. The examples below therefore keep `soffset=0` and put the
complete window-relative offset in `voffset`. For an access that the range check
classifies as OOB, the ISA guarantees:

> An out-of-bounds **read returns 0** (or `1.0` for components selected by
> `dst_sel = SEL_1`); an out-of-bounds **write is dropped**.

That is the whole trick. If you *size the buffer descriptor to the real tensor
extent*, then any lane whose index runs past the end reads a clean `0` and any
overhanging store silently evaporates — no branch, no mask, no clamp, and no
illegal address fault. This is exactly why the AMD compiler's "buffer ops" passes
prefer MUBUF for in-bounds-uncertain accesses. Contrast with
[`global_load`/`flat`](../hardware/memory-instructions.md), which take a raw
64-bit pointer and have **no object-size guard**. An invalid flat address that
does not fall in a memory aperture reports `MEM_VIOL`, but a C/C++ object OOB
address can still land in mapped memory and read or overwrite a neighboring
object. Those paths therefore still need software predication.

## Building the descriptor in HIP

You rarely hand-assemble a V#. A compiler-facing way to get OOB-guarded loads in
HIP is the `__builtin_amdgcn_make_buffer_rsrc` + `__builtin_amdgcn_raw_buffer_load`
intrinsics (a.k.a. the LLVM `amdgcn.raw.ptr.buffer.load` family). Size the
descriptor with the number of **bytes** of valid data:

```cpp
#include <hip/hip_runtime.h>
#include <stdint.h>

// SRD word 3 is architecture-specific. These values match CK Tile's
// CK_TILE_BUFFER_RESOURCE_3RD_DWORD for the targets covered by this page.
#if !defined(__HIP_DEVICE_COMPILE__) || !__HIP_DEVICE_COMPILE__
#define WIKI_RAW_BUFFER_FLAGS 0xffffffffu  // host-pass placeholder; never issued
#elif defined(__gfx942__) || defined(__gfx950__)
#define WIKI_RAW_BUFFER_FLAGS 0x00020000u  // active GFX9 targets
#else
#error "Add the target's raw-buffer resource flags before using this helper"
#endif

// Branchless guarded load inside one <=4-GiB raw-buffer window.
__device__ inline float guarded_load(const float* window_base,
                                     uint32_t byte_offset,
                                     uint32_t valid_bytes)
{
    // make_buffer_rsrc(ptr, stride/*=0 for raw*/, num_records_bytes, flags)
    __amdgpu_buffer_rsrc_t rsrc = __builtin_amdgcn_make_buffer_rsrc(
        const_cast<float*>(window_base),
        /*stride=*/   0,
        /*num_records=*/ valid_bytes,
        /*flags=*/    WIKI_RAW_BUFFER_FLAGS);

    // The b32 builtin returns the raw dword bits; reinterpret them as float.
    int raw = __builtin_amdgcn_raw_buffer_load_b32(
        rsrc,
        /*voffset=*/ byte_offset,
        /*soffset=*/ 0,
        /*aux=*/     0);   // lane past num_records -> hardware returns 0.0f
    return __builtin_bit_cast(float, raw);
}

#undef WIKI_RAW_BUFFER_FLAGS
```

Every lane runs the *same* code; with the shown zero instruction offset and
`soffset`, lanes whose `byte_offset >= valid_bytes` simply receive `0.0f`. No
`if`, no `EXEC` divergence, no separate tail loop. Keep `valid_bytes` a multiple
of the dword component size for this `b32` helper. A single raw descriptor only
checks the resulting **linear byte offset**: it catches a suffix past the end of
that window, but it cannot detect a column overrun that merely lands in the next
mapped row. Guard that dimension separately or describe each row with its own
window when row boundaries matter.

### Allocations larger than the descriptor window

The raw descriptor extent is 32-bit, so one byte-addressed V# covers at most
approximately **4 GiB**. This does **not** limit the allocation itself. Split a
large tensor into windows and rebase the 64-bit descriptor base:

```cpp
// Host or uniform wave-level setup (conceptual):
//   window_base  = allocation + window_start;       // 64-bit base
//   local_offset = absolute_offset - window_start;  // must fit uint32_t
//   valid_bytes  = min(allocation_bytes-window_start, UINT32_MAX);
// guarded_load(window_base, (uint32_t)local_offset, (uint32_t)valid_bytes);
```

If rebasing/chunking is unsuitable, use a 64-bit global address with a software
predicate. The dangerous pattern is computing `n_elems * sizeof(T)` in a signed
32-bit `int`: it can wrap before the descriptor is constructed and make an
invalid region appear in-bounds.

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
[vectorized loads](vectorized-loads.md): raw `buffer_load_dwordx2/x3/x4`
range-checks each dword component independently. A `dwordx4` whose first two
components are in range and last two are out of range therefore returns the two
valid dwords followed by two zeros; only an access with all four component
offsets OOB is fully zero. (Format loads/stores and atomics instead use an
all-or-nothing range check.) This lets a vectorized tail preserve its valid
prefix without scalarizing. Eliminating the tail loop also shrinks the kernel's
instruction footprint and removes the per-iteration compare from the hot loop.
The Triton AMD backend exposes this via its *buffer-ops* lowering
(`tl.load` with a `mask`/`other=0.0` lowers to guarded `buffer_load` when the
pointer is provably a tensor base plus offset — see
[`blog-triton-amd`](../languages/triton-amd.md)).

## Pitfalls

- **`num_records` is in the descriptor's units.** For a raw buffer it is a byte
  count and the hardware compares the instruction/VGPR byte offset; for a
  structured buffer it is a record count and the hardware compares the VGPR
  index. `stride` forms the structured address. Mixing the two changes the range
  test, not merely its scale.
- **It guards range, not source-language alignment.** On gfx942, dword-or-wider
  raw buffer operations ignore the byte address's low two bits, forcing only
  dword alignment; an 8- or 16-byte access is not specified to fault merely
  because it lacks 8- or 16-byte alignment. That hardware rule does not make a
  misaligned typed C++ `float4*` dereference legal: honor the source type's
  alignment contract, or pass a byte offset to a raw-buffer builtin as above.
- **Vector OOB is component-wise for raw dword ops.** `dwordx2/x3/x4` can return
  or store a valid prefix while zeroing/dropping only the OOB dwords. Do not
  assume a partially crossing vector is handled all-or-nothing.
- **Flat/global paths are unguarded.** If a pass demotes your `buffer_load` to a
  `global_load` (e.g. because the descriptor couldn't be proven loop-invariant),
  the object-boundary protection is gone. A mapped neighboring address can be
  accessed without a fault; inspect the assembly for `buffer_` vs `global_`.
- **`soffset` does not extend the raw guard.** It changes the final address, but
  the raw comparison is against `InstOffset + voffset`. Keep it zero for the
  shown window-relative idiom, or account for it when constructing the SRD base
  and range.
- **Descriptor flags are target-specific.** The active gfx942/gfx950 targets
  use `0x00020000`. Use a maintained target-selected helper before broadening
  this code beyond the active GFX9 scope.
- **Negative offsets wrap.** The comparison is unsigned; a negative `voffset`
  becomes a huge positive value (still OOB → 0), which is usually what you want,
  but do not rely on signed clamping.
- **The 4-GiB raw window is not an allocation cap.** Rebase a descriptor per
  chunk and validate every narrowing conversion; structured-buffer record-count
  and stride-based address formation must be read from the target ISA.

## See also

- [Memory instructions: buffer vs global vs flat](../hardware/memory-instructions.md)
- [Vectorized loads](vectorized-loads.md)
- [s_waitcnt async gating](../hardware/s-waitcnt.md)
- [GCN/CDNA assembly](../languages/gcn-asm.md)

## Sources

- [AMD Instinct MI300/CDNA3 ISA Reference Guide — MUBUF / buffer resource descriptor, out-of-range behavior](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf)
- [AMD Instinct CDNA4 ISA Reference Guide](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-cdna4-instruction-set-architecture.pdf) — gfx950 buffer-resource and MUBUF semantics.
- [LLVM AMDGPU backend — `llvm.amdgcn.raw.ptr.buffer.load` / `make.buffer.rsrc` intrinsics](https://llvm.org/docs/AMDGPUUsage.html)
- [CK Tile target configuration — architecture-selected buffer-resource word 3](https://github.com/ROCm/rocm-libraries/blob/5840fddc0f6f42cbedd9ecc113376d760bc177b1/projects/composablekernel/include/ck_tile/core/config.hpp)
- [gcnasm — worked buffer_load examples](https://github.com/AMD/gcnasm)
- [Triton AMD backend — buffer-ops lowering for masked loads](https://rocm.blogs.amd.com/artificial-intelligence/triton/README.html)
- [AMDGPU Kernel Optimization Guide (captured snapshot)](https://github.com/nod-ai/amd-shark-ai/blob/efa471aeef66a260c85983cc41e833bfa769dade/docs/amdgpu_kernel_optimization_guide.md) — raw-buffer predication and the descriptor-window warning.
