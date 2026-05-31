---
id: migration-wmma-vs-mfma
title: 'WMMA (RDNA4) vs MFMA (CDNA3/4): porting matrix-core kernels'
type: migration
architectures:
- gfx942
- gfx950
- gfx1201
tags:
- wmma
- mfma
- matrix-core
- wave32
- wave64
- agpr
- rdna
- cdna
confidence: source-reported
cross_vendor_note: 'AMD exposes two distinct matrix-core ISAs — WMMA on RDNA (consumer/workstation
  gfx11xx/gfx12xx) and MFMA on CDNA (datacenter gfx9xx). This mirrors NVIDIA''s split
  between the `mma.sync`/`nvcuda::wmma` fragment path on consumer GeForce and the
  `wgmma`/`tcgen05` warp-group path on datacenter Hopper/Blackwell. AMD''s WMMA is
  the closest analog to NVIDIA''s `nvcuda::wmma` (warp/wave-level 16×16×16 fragments);
  AMD''s MFMA is the closest analog to the larger tensor-core `mma`/`wgmma` ops. The
  portable abstraction across both AMD families is rocWMMA, analogous in spirit to
  CUTLASS/`nvcuda::wmma`.

  '
related:
- hw-mfma
- hw-wmma
- lang-rocwmma
- migration-cuda-to-hip
- migration-gfx942-to-gfx950
sources:
- doc-llvm-amdgpu
- ref-rocwmma
- blog-amd-matrix-cores
- doc-cdna3-isa
- ref-matrix-calculator
implemented_by:
- pr-composable_kernel-2704
- pr-aiter-3236
- pr-composable_kernel-2528
- pr-composable_kernel-2466
- pr-triton-358
- pr-composable_kernel-3479
- pr-composable_kernel-2110
- pr-FlyDSL-250
---
# WMMA (RDNA4) vs MFMA (CDNA3/4): porting matrix-core kernels

## Why there are two instruction families

AMD ships two unrelated matrix-core instruction sets:

- **MFMA** (`v_mfma_*`, "Matrix Fused Multiply-Add") on **CDNA** datacenter
  parts — gfx942 (MI300) and gfx950 (MI350). See [hw-mfma](../hardware/mfma.md).
- **WMMA** (`v_wmma_*`, "Wave Matrix Multiply-Accumulate") on **RDNA** —
  gfx1100 (RDNA3) and gfx1201 (RDNA4). See [hw-wmma](../hardware/wmma.md).

They are *not* the same encoding, do not share builtins, and have different
register layouts. A kernel hand-tuned with `__builtin_amdgcn_mfma_*` will not
compile for gfx1201, and a `__builtin_amdgcn_wmma_*` kernel will not compile for
gfx942. The portable path is [rocWMMA](../languages/rocwmma.md), whose fragment
API lowers to whichever family the target supports.

## Side-by-side

| Property | MFMA (gfx942 / gfx950) | WMMA (gfx1201) |
|---|---|---|
| Family | CDNA matrix core | RDNA matrix core |
| Mnemonic | `v_mfma_*` (VOP3P-MAI) | `v_wmma_*` (VOP3P) |
| Cooperating lanes | wave64 only | wave32 **or** wave64 |
| Accumulator regs | ArchVGPR **or** AGPR | ArchVGPR only (no AGPRs on RDNA) |
| Canonical shape | 16×16×16, 32×32×8 (and larger-K) | 16×16×16 |
| Large-K / unified low-prec | 16×16×128 `f8f6f4` (gfx950) | not present |
| MX block scaling | `v_mfma_scale_*` (gfx950) | not present |
| Builtin | `__builtin_amdgcn_mfma_*` | `__builtin_amdgcn_wmma_*` |
| Portable wrapper | rocWMMA | rocWMMA |

> The biggest structural differences when porting are (1) **wave width** — RDNA
> can run wave32, CDNA is wave64-only — and (2) **AGPRs** — RDNA has no separate
> accumulator register bank, so all WMMA operands live in ArchVGPRs.

## Wave width: the first thing to fix

CDNA is **wave64-only**; every MFMA is issued by 64 lanes. RDNA supports both
**wave32 and wave64**, and the WMMA register layout differs between the two
modes. Code that hardcodes `64` for lane math, ballot widths, or fragment
strides will silently break on gfx1201 in wave32. Always query the wave size:

```cpp
// Portable across CDNA (always 64) and RDNA (32 or 64 depending on launch).
__device__ int lane_id() {
    return __lane_id();              // 0..warpSize-1
}

// HIP host/device: do NOT assume 64.
//   warpSize == 64 on gfx9 (CDNA), 32 by default on gfx10+/gfx11+/gfx12 (RDNA)
// __ballot returns a 64-bit mask on AMD regardless; mask the unused half in wave32.
```

For matrix code specifically: a 16×16 WMMA fragment is spread over 32 lanes in
wave32 and replicated/repacked differently in wave64. Do not port MFMA
register-packing assumptions by hand — regenerate the layout (see below).

## Accumulators: AGPRs vs ArchVGPRs

On CDNA, the matrix core reads/writes accumulators through a **separate AGPR
bank** (up to 256 AGPRs + 256 ArchVGPRs per wave); accumulator tiles are
conventionally pinned in AGPRs to free ArchVGPRs for addressing. **RDNA has no
AGPRs** — WMMA accumulators occupy ordinary ArchVGPRs. Consequences when porting:

- VGPR-budget math changes: on RDNA the C/D tile competes with A/B fragments and
  address registers for the *same* 256-entry ArchVGPR file, so occupancy tuning
  that relied on the AGPR/ArchVGPR split (see
  [vgpr-budgeting](../techniques/vgpr-budgeting.md)) must be redone.
- No `agpr-management` step exists on RDNA; remove AGPR-to-ArchVGPR copy
  scheduling.

## Shapes and dtypes

MFMA offers a wide shape menu that grows with K for narrow types — e.g. FP16
`16×16×16` and `32×32×8`, FP8 `16×16×32`, and on gfx950 the unified
`v_mfma_f32_16x16x128_f8f6f4` plus MX-scaled variants (see
[hw-mxfp](../hardware/mxfp.md)). WMMA centers on the **16×16×16** fragment shape
across supported dtypes; there is no `32×32` MFMA-style tile and no `f8f6f4`
unified low-precision MMA or `v_mfma_scale_*` block-scaling path on gfx1201.

Practical porting rule: a CDNA kernel that tiles with `32×32×8` MFMA must be
re-tiled to 16×16-based fragments for WMMA, changing the per-wave macro-tile and
therefore the LDS staging and bank-conflict layout.

```cpp
// Conditional matrix-core core, written once with a target switch.
// Real builtins differ per family; pick at compile time by arch macro.
#if defined(__gfx942__) || defined(__gfx950__)
  // CDNA wave64 MFMA: 16x16x16 FP16 -> FP32, accumulator in (Arch/Acc)VGPRs.
  acc = __builtin_amdgcn_mfma_f32_16x16x16f16(a_frag, b_frag, acc, 0, 0, 0);
#elif defined(__gfx1201__) || defined(__gfx1100__)
  // RDNA WMMA: 16x16x16 FP16 -> FP32, accumulator in ArchVGPRs.
  // (wave32 or wave64 — fragment layout follows the launch wave size)
  acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a_frag, b_frag, acc);
#else
  #error "no matrix core for this target"
#endif
```

## Recommended: don't hand-port — use rocWMMA

[rocWMMA](../languages/rocwmma.md) is a header-only C++ fragment API
(`load_matrix_sync` → `mma_sync` → `store_matrix_sync`) that emits `v_mfma_*` on
CDNA (gfx908/90a/942/950) and `v_wmma_*` on RDNA (gfx1100/gfx1201). Writing to
the fragment API once gets you both families, and the library owns the
wave-width- and AGPR-specific register layout you would otherwise hand-derive:

```cpp
#include <rocwmma/rocwmma.hpp>
using namespace rocwmma;

// 16x16x16 tile portable across gfx942 (MFMA) and gfx1201 (WMMA).
__global__ void hgemm_tile(const half* A, const half* B, float* D, int lda, int ldb, int ldd) {
    fragment<matrix_a, 16, 16, 16, half, row_major> fragA;
    fragment<matrix_b, 16, 16, 16, half, col_major> fragB;
    fragment<accumulator, 16, 16, 16, float>        fragAcc;

    fill_fragment(fragAcc, 0.0f);
    load_matrix_sync(fragA, A, lda);
    load_matrix_sync(fragB, B, ldb);
    mma_sync(fragAcc, fragA, fragB, fragAcc);   // -> v_mfma_* or v_wmma_*
    store_matrix_sync(D, fragAcc, ldd, mem_row_major);
}
```

To inspect the *exact* per-lane register mapping for either family (so you can
verify a hand-written layout), use the official AMD Matrix Instruction
Calculator ([ref-matrix-calculator](../../sources/refs/ref-matrix-calculator.md)):

```bash
# CDNA MFMA layout
python3 matrix_calculator.py --architecture cdna3 \
    --instruction v_mfma_f32_16x16x16_f16 --detail-instruction
# RDNA WMMA layout (note wave32 vs wave64 variants)
python3 matrix_calculator.py --architecture rdna4 \
    --instruction v_wmma_f32_16x16x16_f16 --detail-instruction
```

## Porting checklist (CDNA ⇄ RDNA)

1. **Wave size**: stop assuming 64. Query `warpSize`/`__lane_id`; mask ballots
   to the active half in wave32. RDNA fragment layout depends on the launch
   wave mode.
2. **Registers**: drop the AGPR/ArchVGPR split on RDNA; redo VGPR budgeting and
   occupancy against a single ArchVGPR file.
3. **Tile shape**: re-tile `32×32`/large-K MFMA macro-tiles onto 16×16 WMMA
   fragments; revisit LDS staging and bank-conflict avoidance for the new tile.
4. **Low precision**: there is no `f8f6f4` unified op or `v_mfma_scale_*` MX
   path on gfx1201 — fold block scaling into surrounding ALU or pick a supported
   WMMA dtype.
5. **Prefer rocWMMA** for anything that must run on both families; reserve raw
   builtins/asm for a single, profiled target.

## Sources

- [LLVM AMDGPU User Guide — `llvm.amdgcn.mfma.*` / `llvm.amdgcn.wmma.*` intrinsics](https://llvm.org/docs/AMDGPUUsage.html)
- [rocWMMA — portable matrix-core C++ API](https://github.com/ROCm/rocWMMA)
- [AMD Matrix Cores (programming model overview)](https://rocm.blogs.amd.com/software-tools-optimization/amd-matrix-cores/README.html)
- [CDNA3 ISA Reference Guide (MFMA encoding)](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf)
- [AMD Matrix Instruction Calculator](https://github.com/ROCm/amd_matrix_instruction_calculator)
