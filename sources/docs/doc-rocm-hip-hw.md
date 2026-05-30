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
four SIMD16 units; up to 512 total VGPRs per wave (256 Arch + 256 Acc). Up to
40 waves/CU occupancy (four pools × 10 waves), typically limited by register and
LDS usage. CDNA is wave64-only.

## Reference

- Upstream: <https://rocm.docs.amd.com/projects/HIP/en/latest/understand/hardware_implementation.html>
