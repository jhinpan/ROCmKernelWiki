---
id: doc-flash-attention-2
title: 'FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning'
url: https://arxiv.org/abs/2307.08691
source_category: paper
architectures:
- gfx942
- gfx950
tags:
- flash-attention
- attention
- software-pipelining
retrieved_at: '2026-05-15'
---

# FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning

The FlashAttention-2 paper (Tri Dao, 2023). The tiling/online-softmax
algorithm that ROCm's CK-tile and Triton FMHA kernels implement on CDNA. Used here
as the algorithmic reference for the attention kernel pages; the AMD-specific work
(MFMA layout, LDS double-buffering, direct-to-LDS) is layered on top.

## Reference

- Upstream: <https://arxiv.org/abs/2307.08691>
