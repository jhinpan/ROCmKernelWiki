---
id: ref-composable-kernel
title: Composable Kernel (CK / CK-tile)
repo: ROCm/composable_kernel
url: https://github.com/ROCm/composable_kernel
author: ROCm
source_category: reference-repo
architectures:
- gfx942
- gfx950
tags:
- composable-kernel
- gemm
- flash-attention
- fused-moe
languages:
- composable-kernel
- hip
- cpp
retrieved_at: '2026-05-15'
---

# Composable Kernel (CK / CK-tile)

Composable Kernel is a HIP C++ tile-based programming model for
performance-critical ML kernels, built on tensor coordinate-transformation. The
newer `ck_tile` DSL is self-contained (single-header components like
`ck_tile/core.hpp`, `ck_tile/ops/fmha.hpp`) and organizes operators by execution
level: warp → block → pipeline → kernel. Core abstractions: tensor descriptors,
distributed tensors (storage + thread distribution), and tile APIs `load_tile`,
`store_tile`, `shuffle_tile`, `slice_tile`. CK is the primary kernel backend for
many ROCm ML ops. MIT. (Active development has moved to ROCm/rocm-libraries.)

## Reference

- Upstream: <https://github.com/ROCm/composable_kernel>
