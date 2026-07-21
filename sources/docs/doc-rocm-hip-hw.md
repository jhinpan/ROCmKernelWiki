---
id: doc-rocm-hip-hw
title: ROCm HIP — Hardware Implementation
url: https://rocm.docs.amd.com/projects/HIP/en/latest/understand/hardware_implementation.html
source_category: official-doc
architectures:
- gfx942
- gfx950
tags:
- wave64
- sgpr
- vgpr
- agpr
- cu
- cdna
retrieved_at: '2026-05-15'
---

# ROCm HIP — Hardware Implementation

ROCm HIP hardware-implementation reference. Documents the CU register
files: ~12.5 KiB SGPR storage per CU; 256–512 KiB VGPR storage split across the
four SIMD16 units; up to 256 ArchVGPR and 256 AGPR names sharing one combined
512-entry-per-lane allocation on gfx942/gfx950. Up to
40 waves/CU is a generic/older-GCN limit described by this upstream overview;
it is **not** the gfx942/gfx950 limit. CDNA3/CDNA4 devices report 32 waves/CU
(four SIMD pools × 8 waves), typically reduced further by register and LDS
usage. CDNA is wave64-only. See [`hw-wavefront`](../../wiki/hardware/wavefront.md)
for the device-verified architecture-scoped value.

## Reference

- Upstream: <https://rocm.docs.amd.com/projects/HIP/en/latest/understand/hardware_implementation.html>
