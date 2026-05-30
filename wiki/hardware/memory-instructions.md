---
id: hw-memory-instructions
title: "Memory Instructions — buffer (MUBUF) vs global vs flat"
type: hardware
architectures:
- gfx942
- gfx950
- gfx1201
tags:
- buffer-instructions
- global-instructions
- s-waitcnt
- buffer-oob-guard
- vectorized-loads
- vgpr
confidence: verified
evidence_basis:
- source_id: doc-cdna3-isa
  evidence_type: official-doc
- source_id: doc-llvm-amdgpu
  evidence_type: official-doc
- source_id: ref-gcnasm
  evidence_type: upstream-code
related:
- hw-async-copy-lds
- hw-s-waitcnt
- hw-lds
- technique-buffer-oob-guard
- technique-vectorized-loads
- lang-gcn-asm
sources:
- doc-cdna3-isa
- doc-llvm-amdgpu
- ref-gcnasm
- doc-rocm-hip-hw
- blog-gemm-optimization
aliases:
- MUBUF
- buffer_load
- global_load
- flat_load
- "V#"
- "resource descriptor"
---

# Memory Instructions — buffer (MUBUF) vs global vs flat

## Overview

CDNA/RDNA cores reach memory (HBM, the L2/L1 cache hierarchy, scratch, and LDS)
through three distinct vector-memory instruction classes. They differ in how
they *form an address* and in what *out-of-bounds (OOB)* means — and that
difference is something a kernel engineer exploits deliberately, not an
implementation detail to ignore.

| Class | Mnemonics | Address model | OOB semantics | Counters |
|---|---|---|---|---|
| **buffer (MUBUF)** | `buffer_load_*`, `buffer_store_*` | 128-bit resource descriptor (V#) + 32-bit VGPR offset (+ optional SGPR `soffset`/`inst_offset`) | **read → 0, write → dropped** (per `num_records`) | VMCNT |
| **global** | `global_load_*`, `global_store_*` | flat 64-bit address, *or* SGPR base (`saddr`) + 32-bit VGPR offset | page fault / `MEM_VIOL` | VMCNT only |
| **flat** | `flat_load_*`, `flat_store_*` | single 64-bit address resolved at runtime to global / LDS / scratch | page fault / `MEM_VIOL` | **VMCNT *and* LGKMCNT** |

The picks in practice:

- **buffer** when you have a base pointer + a structured offset and want
  *free, branchless bounds checking* (GEMM/attention tiles, ragged batches).
- **global** when you have a raw 64-bit pointer and bounds are guaranteed (or
  guarded in software) — the common case for plain HIP pointer arithmetic.
- **flat** only when the address space is genuinely unknown at compile time;
  it is the most general and the most expensive to wait on.

## The 128-bit resource descriptor (V#)

A MUBUF instruction does not take a pointer; it takes a **buffer resource
descriptor**, four consecutive SGPRs (128 bits), conventionally called a *V#*.
The descriptor packs the base address, a per-lane/record **stride**, a
**num_records** field that defines the in-bounds window, and format/swizzle
control bits. The hardware computes the effective address from the V# plus the
instruction's VGPR offset and compares it against `num_records` to decide
in-bounds vs OOB.

```cpp
// Build a V# in C++ and issue a bounds-checked vectorized load.
// make.buffer.rsrc packs {base_ptr, stride, num_records, flags} -> a __amdgpu_buffer_rsrc_t.
__device__ float4 load_tile_guarded(const float* base, int num_elems, int idx) {
    // 4-dword (128-bit) descriptor: stride=0 (raw byte buffer),
    // num_records = byte extent -> anything past it reads as 0.
    __amdgpu_buffer_rsrc_t rsrc =
        __builtin_amdgcn_make_buffer_rsrc(base,
                                          /*stride   */ (short)0,
                                          /*num_recs */ num_elems * (int)sizeof(float),
                                          /*flags    */ 0);
    // byte offset for this lane; if idx is OOB the load returns 0.0f, no branch.
    int byte_off = idx * (int)sizeof(float);
    return __builtin_amdgcn_raw_buffer_load_b128(rsrc, byte_off, /*soffset*/ 0, /*aux*/ 0);
}
```

The same descriptor in standalone assembly (the form the Tensile/CK backends
emit) — load four SGPRs, then issue the MUBUF op:

```asm
; s[0:1] = base pointer, s2 = num_records (bytes), v0 = per-lane byte offset
s_mov_b32      s4, s0           ; SRD[0] = base_lo
s_mov_b32      s5, s1           ; SRD[1] = base_hi
s_mov_b32      s6, s2           ; SRD[2] = num_records
s_mov_b32      s7, 0x00020000   ; SRD[3] = flags (raw, 32-bit dword element)
buffer_load_dwordx4 v[8:11], v0, s[4:7], 0 offen   ; OOB lanes -> 0
s_waitcnt      vmcnt(0)
```

## OOB semantics: read = 0, write = dropped

This is the headline property. For a MUBUF access whose computed record index is
**≥ num_records** (or otherwise out of the descriptor window):

- a **load returns 0** for that lane (or `1` when `dst_sel = SEL_1` is selected
  for a component), and
- a **store is silently dropped** for that lane.

No fault, no divergence, no per-lane branch. This turns the classic boundary
check at the edge of a tile into *nothing* — you size the V# to the real tensor
extent and let the hardware mask the ragged lanes. It is the foundation of the
[branchless buffer OOB guard](../techniques/buffer-oob-guard.md) and is why
GEMM/attention kernels that must handle M/N/K not divisible by the tile size
prefer buffer ops over `global` for the load path. The cost of an OOB lane is
that it produces a zero rather than useful data — the instruction slot is still
spent — so it is a correctness/simplicity tool, not a way to skip work.

> **global has no such safety net.** A `global_load`/`global_store` to an
> unmapped address faults (`MEM_VIOL`); there is no `num_records` clamp. If you
> use `global` at a tensor edge you must guard with an `EXEC`-mask / predicated
> branch yourself. That is the trade: `global` saves the 4-SGPR descriptor setup,
> buffer ops save the boundary branch.

## global: lean flat-ish addressing

`global_load_*`/`global_store_*` use either a full 64-bit address spread across a
VGPR pair, or — more efficiently — a **scalar base (`saddr`) plus a 32-bit VGPR
offset**, which keeps the per-lane address in a single VGPR and the common base
in SGPRs. Global ops touch only global memory; **issuing one against an LDS
address is illegal and raises `MEM_VIOL`**. Because they never touch LDS or
scalar/const memory, they increment **only VMCNT** — so a downstream
`s_waitcnt vmcnt(N)` is enough, which makes software pipelining cleaner than with
flat. See [s_waitcnt](s-waitcnt.md) for the counter model.

```cpp
// Plain HIP pointer math compiles to global_load/global_store (saddr + voffset).
__global__ void saxpy(const float* __restrict__ x, float* __restrict__ y, float a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)                          // explicit guard: global has no OOB clamp
        y[i] = a * x[i] + y[i];         // -> global_load_b32 / global_store_b32
}
```

## flat: the general, expensive case

A `flat_*` instruction carries a single 64-bit address whose **aperture** (which
address space it lands in) is resolved *at runtime* — it may be global, LDS, or
scratch. Because a flat op can touch both VMEM and LDS, it bumps **both VMCNT and
LGKMCNT**, and those two counter classes can retire out of order. The practical
consequence: after a flat op there is no safe partial wait, so only
`s_waitcnt vmcnt(0) lgkmcnt(0)` (i.e. `s_waitcnt 0`) is meaningful. Prefer
`global`/`buffer` whenever the address space is statically known; reserve flat
for generic pointers (e.g. `__attribute__((address_space(0)))` traffic the
compiler cannot prove).

## scratch: per-lane private spill

Scratch is the private memory backing register spills, large local arrays, and
the call stack. It is addressed per-lane via the dedicated `scratch_load_*` /
`scratch_store_*` ops (or implicitly when the compiler spills VGPRs). Scratch is
physically carved out of the same HBM/L2 path as global memory, with a private
aperture and per-wave swizzling so consecutive lanes hit consecutive addresses.
Scratch traffic is a *symptom*, not a goal: if `rocprof`/`rocprofv3` shows
scratch loads/stores in a hot kernel you are
[spilling registers](../patterns/vgpr-pressure.md) — cut VGPR/AGPR pressure
rather than optimizing the scratch path.

## Vectorization and waits

All three classes have width variants — `b32`, `b64`, `b96`/`dwordx3`,
`b128`/`dwordx4`. Issuing the widest access the alignment allows
(`buffer_load_dwordx4` / `global_load_b128`, 16 bytes/lane) minimizes
instruction count and maximizes in-flight bytes per VMCNT slot; see
[vectorized loads](../techniques/vectorized-loads.md). For staging into LDS
without burning VGPRs, the **direct-to-LDS** variants (`buffer_load_dword ... lds`
/ `global_load_lds_*`) move HBM→LDS bypassing the register file — covered on the
[async-copy-to-LDS page](async-copy-lds.md).

## See also

- [Direct-to-LDS async copy](async-copy-lds.md)
- [s_waitcnt counter model](s-waitcnt.md)
- [Branchless buffer OOB guard](../techniques/buffer-oob-guard.md)
- [Vectorized loads](../techniques/vectorized-loads.md)
- [GCN/CDNA assembly](../languages/gcn-asm.md)

## Sources

- [AMD Instinct MI300 / CDNA3 Instruction Set Architecture](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf) — MUBUF / FLAT / GLOBAL / SCRATCH formats, buffer resource descriptor, OOB semantics.
- [LLVM AMDGPU Backend User Guide](https://llvm.org/docs/AMDGPUUsage.html) — `make.buffer.rsrc`, `raw.buffer.load/store` intrinsics, address spaces.
- [AMD ISA assembly examples (gcnasm)](https://github.com/AMDResearch) — standalone MUBUF descriptor setup and buffer-op idioms.
- [ROCm HIP hardware programming guide](https://rocm.docs.amd.com/) — memory address spaces and pointer mapping.
