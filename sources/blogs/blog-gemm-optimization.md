---
id: blog-gemm-optimization
title: GEMM Kernel Optimization for AMD GPUs
author: AMD ROCm
url: https://rocm.blogs.amd.com/artificial-intelligence/gemm_blog/README.html
source_category: benchmark-blog
architectures:
- gfx942
tags:
- gemm
- mfma
- lds-double-buffering
- swizzle
- tile-scheduling
retrieved_at: '2026-05-15'
---

# GEMM Kernel Optimization for AMD GPUs

End-to-end walkthrough of GEMM optimization on MI300: macro-tiling,
LDS staging with double buffering, MFMA scheduling, bank-conflict avoidance via
swizzled LDS layouts, and work-group → L2 tile mapping for locality.

## Reference

- Upstream: <https://rocm.blogs.amd.com/artificial-intelligence/gemm_blog/README.html>
