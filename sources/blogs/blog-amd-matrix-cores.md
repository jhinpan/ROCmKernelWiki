---
id: blog-amd-matrix-cores
title: AMD Matrix Cores
author: AMD ROCm
url: https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores/README.html
source_category: benchmark-blog
architectures:
- gfx942
tags:
- mfma
- matrix-core
- bf16
- fp16
retrieved_at: '2026-05-15'
---

# AMD Matrix Cores

The foundational ROCm blog on AMD Matrix Cores. Introduces the
`__builtin_amdgcn_mfma_*` intrinsics, the per-wavefront register-fragment layout
for A/B/C/D operands, and the `v_mfma_f32_16x16x16f16` / `v_mfma_f32_32x32x8f16`
shapes. The starting point for understanding MFMA programming.

## Reference

- Upstream: <https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores/README.html>
