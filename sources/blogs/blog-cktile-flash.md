---
id: blog-cktile-flash
title: 'From Theory to Kernel: Implement FlashAttention-v2 with CK-Tile'
author: AMD ROCm
url: https://rocm.blogs.amd.com/software-tools-optimization/ck-tile-flash/README.html
source_category: benchmark-blog
architectures:
- gfx942
tags:
- flash-attention
- attention
- mfma
- lds-double-buffering
retrieved_at: '2026-05-15'
---

# From Theory to Kernel: Implement FlashAttention-v2 with CK-Tile

Implements FlashAttention-2 with CK-tile: online-softmax tiling,
back-to-back MFMA GEMMs (QK^T then PV), LDS staging, and the block/pipeline
operator structure for a fused attention kernel on MI300.

## Reference

- Upstream: <https://rocm.blogs.amd.com/software-tools-optimization/ck-tile-flash/README.html>
