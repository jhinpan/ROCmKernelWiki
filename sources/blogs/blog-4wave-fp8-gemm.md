---
id: blog-4wave-fp8-gemm
title: Deep Dive Into 4-Wave Interleave FP8 GEMM
author: AMD ROCm
url: https://rocm.blogs.amd.com/software-tools-optimization/4wave-fp8gemm/README.html
source_category: benchmark-blog
architectures:
- gfx950
tags:
- fp8-gemm
- mfma-pipelining
- occupancy-tuning
- wave-specialization
retrieved_at: '2026-05-15'
---

# Deep Dive Into 4-Wave Interleave FP8 GEMM

A detailed case study of a 4-wave interleaved FP8 GEMM on CDNA4,
overlapping MFMA issue across waves to hide latency and saturate the matrix cores.

## Reference

- Upstream: <https://rocm.blogs.amd.com/software-tools-optimization/4wave-fp8gemm/README.html>
