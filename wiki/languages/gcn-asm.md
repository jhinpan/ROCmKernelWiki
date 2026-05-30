---
id: lang-gcn-asm
title: "GCN/CDNA Assembly — Registers, Mnemonics, and Inline ASM"
type: language
architectures:
- gfx942
- gfx950
- gfx1201
tags:
- gcn-asm
- sgpr
- vgpr
- agpr
- s-waitcnt
- buffer-instructions
- ds-instructions
- mfma
- wave64
confidence: source-reported
reproducibility: snippet
languages:
- gcn-asm
- hip
related:
- lang-hip
- hw-memory-instructions
- hw-s-waitcnt
- hw-mfma
- hw-cross-lane
- kernel-vector-add-asm
sources:
- doc-cdna3-isa
- doc-llvm-amdgpu
- ref-gcnasm
- hw-memory-instructions
- hw-s-waitcnt
aliases:
- GCN assembly
- CDNA assembly
- AMDGCN asm
- "amdgcn ISA"
---

# GCN/CDNA Assembly — Registers, Mnemonics, and Inline ASM

## Overview

GCN/CDNA assembly is the lowest level a kernel engineer normally touches on AMD
Instinct GPUs. The same instruction set underlies everything the higher layers
emit — HIP/Clang, [Composable Kernel](composable-kernel.md), Tensile, and the
Triton AMD backend all bottom out in `v_mfma_*`, `buffer_load_*`, `ds_read_*`,
and `s_waitcnt`. Reading this assembly is essential when debugging register
spills, verifying that the compiler issued the wide vector loads you expected, or
hand-writing a hot inner loop (Tensile's GEMM kernels are literally generated
assembly).

This page covers the register model, the mnemonic families you will see most,
the structure of a standalone `.s` kernel, and how to drop inline assembly into
HIP. CDNA (gfx9xx) is **wave64-only**; RDNA4 (gfx1201) shares most of the scalar
and memory ISA but uses `v_wmma_*` instead of `v_mfma_*` and can run wave32.

## Register model

| Class | Name | Notes |
|---|---|---|
| Scalar | `s0`, `s1`, … (SGPR) | One value per wavefront (uniform). ~12.5 KiB/CU. Holds descriptors, kernel args, loop counters. |
| Vector | `v0`, `v1`, … (ArchVGPR) | One value per lane (64 lanes). Up to 256/wave. |
| Accumulator | `a0`, `a1`, … (AGPR) | MFMA accumulators. Up to 256/wave (512 total VGPRs Arch+Acc). |
| `exec` | 64-bit execute mask | Bit `i` gates lane `i`; predication is masking, not branching. |
| `vcc` | vector condition code | Per-lane carry/compare results. |
| `m0` | memory/index reg | Must be initialized before some LDS ops. |

Multi-dword operands use bracket ranges: `s[0:3]` is a 128-bit scalar group (e.g.
a buffer resource descriptor, "V#"), `v[8:11]` is four contiguous VGPRs holding
an MFMA accumulator tile. VGPRs are allocated in groups of 8 dwords, so register
*budget* is quantized — relevant to [occupancy](../patterns/vgpr-pressure.md).

## Mnemonic families

**Memory** (see [memory instructions](../hardware/memory-instructions.md)):

```asm
; MUBUF: typed/untyped buffer access through a 128-bit V# in s[0:3].
; Out-of-bounds reads return 0 and OOB writes are dropped -> branchless guards.
buffer_load_dwordx4  v[4:7], v0, s[0:3], 0 offen   ; load 16 bytes, per-lane offset in v0
buffer_store_dwordx4 v[4:7], v0, s[0:3], 0 offen

; FLAT/global: 64-bit addressing, no descriptor needed.
global_load_dwordx4  v[4:7], v[0:1], off           ; address pair in v[0:1]
global_store_dwordx4 v[0:1], v[4:7], off

; Direct-to-LDS async copy: HBM -> LDS, bypassing VGPRs (AMD's cp.async analog).
buffer_load_dword    v0, s[0:3], 0 offen lds
```

**LDS / cross-lane** (see [cross-lane](../hardware/cross-lane.md)):

```asm
ds_read_b128   v[8:11], v20            ; 16-byte LDS read
ds_write_b128  v20, v[8:11]
ds_read2_b64   v[8:11], v20 offset0:0 offset1:8   ; strided pair
ds_bpermute_b32 v5, v6, v7             ; 64-lane gather via LDS crossbar (no LDS used)
ds_swizzle_b32  v5, v6 offset:0x1f     ; dword swizzle within 32-lane group
```

**Matrix** ([MFMA](../hardware/mfma.md)) — issued by the whole wavefront, with
accumulators conventionally in AGPRs:

```asm
; D[16x16] += A[16x16] * B[16x16], FP16 in -> FP32 acc, 4 acc VGPRs
v_mfma_f32_16x16x16_f16 a[0:3], v[0:1], v[2:3], a[0:3]
; gfx950 unified low-precision path:
v_mfma_f32_16x16x128_f8f6f4 a[0:3], v[0:7], v[8:15], a[0:3]
```

**Async gating** — [`s_waitcnt`](../hardware/s-waitcnt.md) is how the wave
synchronizes with outstanding memory traffic. There is **no** mbarrier/TMA; the
compiler schedules monotonic per-wave counters:

```asm
buffer_load_dwordx4 v[4:7],  v0, s[0:3], 0 offen   ; VMEM op #1 (VMCNT++)
buffer_load_dwordx4 v[8:11], v1, s[0:3], 0 offen   ; VMEM op #2
ds_read_b128        v[12:15], v20                   ; LDS op   (LGKMCNT++)
; ... independent ALU work overlaps the loads ...
s_waitcnt vmcnt(1)                                  ; wait until only 1 VMEM op outstanding
; v[4:7] is now safe to use; the second load may still be in flight
s_waitcnt lgkmcnt(0)                                ; v[12:15] (LDS) is now safe
```

`vmcnt` counts buffer/flat/global loads/stores; `lgkmcnt` counts LDS + scalar
constant-memory + message ops; `expcnt` counts exports (unused on CDNA4). Same-type
ops retire in order, so `vmcnt(N)` means "all but the N most recent VMEM ops have
completed."

## Anatomy of a standalone `.s` kernel

A kernel object needs an AMDHSA kernel descriptor (`.amdhsa_kernel`) plus a
metadata block describing the arguments. The descriptor declares the register and
LDS footprint the runtime must reserve.

```asm
.amdgcn_target "amdgcn-amd-amdhsa--gfx942"
.text
.globl vadd
.p2align 8
.type vadd,@function
vadd:
    ; s[0:1] = kernarg base pointer (set up by the dispatch)
    s_load_dwordx2 s[4:5], s[0:1], 0x00      ; load arg0 ptr (out)
    s_load_dwordx2 s[6:7], s[0:1], 0x08      ; load arg1 ptr (in a)
    s_load_dwordx2 s[8:9], s[0:1], 0x10      ; load arg2 ptr (in b)
    s_waitcnt lgkmcnt(0)                      ; wait for scalar loads (LGKMCNT)
    v_lshlrev_b32 v0, 2, v0                   ; byte offset = tid * 4
    global_load_dword v1, v0, s[6:7]
    global_load_dword v2, v0, s[8:9]
    s_waitcnt vmcnt(0)                         ; wait for both global loads
    v_add_f32 v1, v1, v2
    global_store_dword v0, v1, s[4:5]
    s_endpgm

.rodata
.p2align 6
.amdhsa_kernel vadd
    .amdhsa_user_sgpr_kernarg_segment_ptr 1
    .amdhsa_next_free_vgpr 3
    .amdhsa_next_free_sgpr 10
    .amdhsa_group_segment_fixed_size 0        ; LDS bytes
.end_amdhsa_kernel
```

Assemble and link with the LLVM toolchain shipped in ROCm:

```bash
clang -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx942 \
      vadd.s -o vadd.o
clang -target amdgcn-amd-amdhsa -mcpu=gfx942 vadd.o -o vadd.hsaco
# Inspect what the compiler/assembler actually produced:
llvm-mc -arch=amdgcn -mcpu=gfx942 -show-encoding vadd.s
roc-obj-ls vadd.hsaco          # list code objects
```

To read assembly out of an existing HIP binary, disassemble the code object:

```bash
hipcc -c --offload-arch=gfx942 -save-temps mykernel.hip   # emits *.s
llvm-objdump -d --mcpu=gfx942 mykernel.o                   # disassemble
```

## Inline assembly in HIP

You rarely need full `.s` files — most low-level needs are met by Clang's
GCN-aware `__builtin_amdgcn_*` intrinsics (preferred, because the register
allocator and scheduler stay in control). Reach for raw inline `asm` only for an
instruction the compiler will not emit or to pin an exact encoding.

```cpp
__device__ float ds_swizzle_xor1(float x) {
    // Prefer the builtin: lets the scheduler insert wait states / track EXEC.
    int xi = __builtin_bit_cast(int, x);
    int y  = __builtin_amdgcn_ds_swizzle(xi, 0x041f); // XOR within 32-lane group
    return __builtin_bit_cast(float, y);
}

__device__ float add_inline(float a, float b) {
    float d;
    // volatile prevents the optimizer from deleting/reordering the op.
    asm volatile("v_add_f32 %0, %1, %2"
                 : "=v"(d)        // %0 -> a VGPR
                 : "v"(a), "v"(b) // %1,%2 from VGPRs
    );
    return d;
}
```

Constraint letters: `v` = VGPR, `s` = SGPR, `a` = AGPR, `=` = write-only,
`+` = read-write. Inline `asm` is opaque to the scheduler, so you are responsible
for any required `s_waitcnt` and for DPP/MFMA hazard wait-states — getting these
wrong yields silent data corruption, not a compile error. When in doubt, use the
builtin and verify with `llvm-objdump`.

## Practical notes

- **`warpSize` is 64 on CDNA** — never hardcode 32. Lane masks (`__ballot`,
  `exec`, `vcc`) are 64-bit. RDNA4 may run wave32.
- The compiler's scheduler hoists loads and packs `s_waitcnt` aggressively;
  if you see a stall, check whether a `vmcnt(0)` is gating more than necessary.
- Read disassembly to confirm vector width: a tight copy loop should show
  `_dwordx4` (16-byte) ops, not scalar `_dword` — see
  [vectorized loads](../techniques/vectorized-loads.md).
- The [AMD Matrix Instruction Calculator](../../sources/refs/ref-matrix-calculator.md)
  resolves which `a[...]`/`v[...]` registers hold each MFMA matrix element.

## Sources

- [AMD Instinct MI300 / CDNA3 ISA Reference Guide](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf)
- [LLVM AMDGPU Backend — User Guide & Inline Asm Constraints](https://llvm.org/docs/AMDGPUUsage.html)
- [GCN/CDNA assembly reference & examples (gcnasm)](https://github.com/RadeonOpenCompute/ROCm)
- [Memory instructions: buffer vs global vs flat](../hardware/memory-instructions.md)
- [s_waitcnt counters and async gating](../hardware/s-waitcnt.md)
