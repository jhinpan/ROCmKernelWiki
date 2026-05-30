---
id: blog-gluon-gemm
title: 'From Naive to Near-Peak: Building High-Performance GEMM Kernels with Gluon'
author: AMD ROCm
url: https://rocm.blogs.amd.com/software-tools-optimization/gluon-gemm-tutorial/README.html
source_category: benchmark-blog
architectures:
- gfx950
tags:
- gemm
- mfma-pipelining
- async-copy
- lds-double-buffering
retrieved_at: '2026-05-15'
---

# From Naive to Near-Peak: Building High-Performance GEMM Kernels with Gluon

A step-by-step GEMM optimization in Gluon (Triton's lower-level
layer), progressing from a naive tiling to a near-peak kernel by adding LDS
buffering, async copy, and MFMA scheduling on CDNA4.

## Reference

- Upstream: <https://rocm.blogs.amd.com/software-tools-optimization/gluon-gemm-tutorial/README.html>
