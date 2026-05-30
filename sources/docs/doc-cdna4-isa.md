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

Key facts used across this wiki:

- The `f8f6f4` ops **repurpose CBSZ/BLGP** as per-matrix element-format selectors:
  `000`=E4M3 (FP8), `001`=E5M2 (BF8), `010`=E2M3 (FP6), `011`=E3M2 (BF6),
  `100`=E2M1 (FP4). Mixed A/B formats are allowed.
- MX scale format is **E8M0**; `ABID[0]=1` enables scaling (else all scales 1.0).
  The hardware folds scales into the exponent sum.
- LDS grows to **160 kB/CU, 64 banks of 640 Dwords**.
- Direct-to-LDS widens to 12/16-byte copies (`GLOBAL_LOAD_LDS_DWORDX3/X4`).
- `v_permlane16_*` cross-lane ops are added (absent on gfx942).
- EXPCNT is "Unused"; TF32/XF32 native path dropped in favor of MX formats.

## Reference

- Upstream: <https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-cdna4-instruction-set-architecture.pdf>
