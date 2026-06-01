---
id: doc-cdna4-isa
title: CDNA4 Instruction Set Architecture Reference Guide
url: https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-cdna4-instruction-set-architecture.pdf
source_category: official-doc
architectures:
- gfx950
tags:
- mfma
- fp8
- fp6
- fp4
- mxfp
- lds
- block-scale
- cdna
retrieved_at: '2026-05-15'
---

# CDNA4 Instruction Set Architecture Reference Guide

The authoritative ISA reference for CDNA4 (gfx950 / MI350-MI355X),
revision 5-August-2025. Adds the unified low-precision matrix instructions
`v_mfma_f32_16x16x128_f8f6f4` and `v_mfma_f32_32x32x64_f8f6f4`, plus their
microscaling (MX) `v_mfma_scale_*` variants.

## Matrix (MFMA) instructions

- Adds the **unified low-precision** dense ops `v_mfma_f32_16x16x128_f8f6f4` and
  `v_mfma_f32_32x32x64_f8f6f4` (FP32 accumulate), plus their microscaling
  variants `v_mfma_scale_f32_16x16x128_f8f6f4` / `v_mfma_scale_f32_32x32x64_f8f6f4`.
- The `f8f6f4` ops **repurpose CBSZ/BLGP** as per-matrix element-format selectors
  (CBSZ = matrix A format, BLGP = matrix B format), mixed A/B allowed:
  `000`=E4M3 (FP8), `001`=E5M2 (BF8), `010`=E2M3 (FP6), `011`=E3M2 (BF6),
  `100`=E2M1 (FP4).
- **MX scale** format is **E8M0** (8-bit exponent, bias 127), one shared scale per
  MX block; `ABID[0]=1` enables scaling (else all scales forced to 1.0). The
  hardware folds the scale into the exponent sum:
  `d_exp = Σ(aᵢ_exp + bᵢ_exp) + c_exp + scale_a + scale_b`.
- Keeps the full CDNA3 MFMA/SMFMAC set and adds wider-K halves
  (`v_mfma_f32_16x16x32_{f16,bf16}`, `v_mfma_f32_32x32x16_{f16,bf16}`,
  `v_mfma_i32_16x16x64_i8`). The native **TF32/XF32** matrix path is **dropped**
  (BF16-emulated) and **FP64 matrix** throughput per CU is **halved** vs CDNA3.
- Companion conversion ops for packing/unpacking the narrow formats with E8M0
  scales: `v_cvt_scalef32_pk_fp4_f32`, `v_cvt_scalef32_pk32_fp6_f32`, plus
  stochastic-rounding `..._sr_...` variants.

## Memory & LDS

- **LDS** grows to **160 kB/CU, 64 banks of 640 Dwords** (32-bit wide) — vs
  64 kB / 32 banks on CDNA3. Bank index is `(address / 4) % 64`.
- **Direct-to-LDS** widens to 12/16-byte copies
  (`GLOBAL_LOAD_LDS_DWORDX3` / `GLOBAL_LOAD_LDS_DWORDX4`); the LLVM intrinsic
  `llvm.amdgcn.load.to.lds` selects these on gfx950.

## Wavefront / counters / cross-lane

- `v_permlane16_*` cross-lane ops are **added** (absent on gfx942) — see the
  latency comparison in `blog-amdgpu-kernel-opt-guide`.
- **EXPCNT** is "Unused"; **LGKMCNT** wording drops GDS. CDNA remains wave64-only.

## Reference

- Upstream: <https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-cdna4-instruction-set-architecture.pdf>
