---
id: blog-fp8-gemm-cdna4
title: FP8 GEMM Optimization on AMD CDNA4 Architecture
author: AMD ROCm
url: https://rocm.blogs.amd.com/software-tools-optimization/cdna4-gemm-kernels/README.html
source_category: benchmark-blog
architectures:
- gfx950
tags:
- fp8-gemm
- fp8
- block-scale
- mfma
- async-copy
retrieved_at: '2026-05-15'
---

# FP8 GEMM Optimization on AMD CDNA4 Architecture

FP8 GEMM tuning specifically for gfx950: using the OCP-FP8 `f8f6f4`
MFMA path, E8M0 block scaling, direct-to-LDS async copy (now 16-byte wide), and
the larger 160 kB LDS to deepen the software pipeline.

## Reference

- Upstream: <https://rocm.blogs.amd.com/software-tools-optimization/cdna4-gemm-kernels/README.html>
