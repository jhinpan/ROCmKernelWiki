---
id: doc-llvm-amdgpu
title: LLVM — User Guide for the AMDGPU Backend
url: https://llvm.org/docs/AMDGPUUsage.html
source_category: official-doc
architectures:
- gfx942
- gfx950
- gfx1201
tags:
- mfma
- rocdl
- async-copy
- block-scale
retrieved_at: '2026-05-15'
---

# LLVM — User Guide for the AMDGPU Backend

The LLVM AMDGPU backend user guide: target names (gfx942, gfx950,
gfx1201), the `llvm.amdgcn.mfma.*` and `llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4`
intrinsics, and `llvm.amdgcn.load.to.lds` (lowers to `global_load_lds` /
`buffer_load_*_lds`; gfx950 allows 12/16-byte copies). Authoritative reference for
the intrinsics ROCm/HIP/Triton ultimately emit.

## Reference

- Upstream: <https://llvm.org/docs/AMDGPUUsage.html>
