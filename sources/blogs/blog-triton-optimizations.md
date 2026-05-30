---
id: blog-triton-optimizations
title: Unlock Peak Performance on AMD GPUs with Triton Kernel Optimizations
author: AMD ROCm
url: https://rocm.blogs.amd.com/software-tools-optimization/kernel-development-optimizations-with-triton-on-/README.html
source_category: benchmark-blog
architectures:
- gfx942
- gfx950
tags:
- mfma-pipelining
- occupancy-tuning
- async-copy
retrieved_at: '2026-05-15'
---

# Unlock Peak Performance on AMD GPUs with Triton Kernel Optimizations

The key Triton-on-AMD tuning knobs: `matrix_instr_nonkdim` (MFMA
size selection), `waves_per_eu` (occupancy), `kpack` (K-packing; deprecated on
gfx950), `num_stages` (pipeline depth), and async-copy / buffer-ops passes.

## Reference

- Upstream: <https://rocm.blogs.amd.com/software-tools-optimization/kernel-development-optimizations-with-triton-on-/README.html>
