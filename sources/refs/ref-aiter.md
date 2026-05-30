---
id: ref-aiter
title: AITER — AI Tensor Engine for ROCm
repo: ROCm/aiter
url: https://github.com/ROCm/aiter
author: ROCm
source_category: reference-repo
architectures:
- gfx942
- gfx950
tags:
- fused-moe
- attention
- mla
- paged-attention
- quantization
languages:
- hip
- cpp
- triton
retrieved_at: '2026-05-15'
---

# AITER — AI Tensor Engine for ROCm

AITER is AMD's high-performance AI operator library — the default
kernel backend for LLM inference on AMD GPUs (e.g. vLLM's default attention
backend). It offers C++ and Python APIs and dispatches across multiple kernel
backends: Triton, Composable Kernel, and hand-tuned assembly (with optional
FlyDSL kernels for mixed-precision MoE, falling back to CK when absent). Coverage:
attention (MHA, MLA, Paged Attention), fused MoE, GEMM, normalization,
quantization, and fused GEMM+communication.

## Reference

- Upstream: <https://github.com/ROCm/aiter>
