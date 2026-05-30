---
id: blog-matrix-cores-cdna
title: Matrix Core Programming on AMD CDNA3 and CDNA4 Architecture
author: AMD ROCm
url: https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores-cdna/README.html
source_category: benchmark-blog
architectures:
- gfx942
- gfx950
tags:
- mfma
- fp8
- fp6
- fp4
- mxfp
- block-scale
retrieved_at: '2026-05-15'
---

# Matrix Core Programming on AMD CDNA3 and CDNA4 Architecture

The definitive intrinsic-level reference for MFMA on gfx942 and
gfx950. Covers the `__builtin_amdgcn_mfma_f32_*` builtins, the CDNA4 `f8f6f4`
unified low-precision ops, scaled MFMA with E8M0 block scales, and the FNUZ-vs-OCP
FP8 distinction between the two architectures.

## Reference

- Upstream: <https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores-cdna/README.html>
