---
id: ref-rocwmma
title: rocWMMA — C++ WMMA-style MMA library
repo: ROCm/rocWMMA
url: https://github.com/ROCm/rocWMMA
author: ROCm
source_category: reference-repo
architectures:
- gfx942
- gfx950
- gfx1201
tags:
- matrix-core
- mfma
- wmma
- gemm
languages:
- cpp
- hip
retrieved_at: '2026-05-15'
---

# rocWMMA — C++ WMMA-style MMA library

rocWMMA is a header-only C++ library for mixed-precision MMA. It
exposes a CUDA-`wmma`-like fragment API (load → mma_sync → store) that compiles
directly into `v_mfma_*` (CDNA) or `v_wmma_*` (RDNA) instructions, handling the
wavefront register-fragment layout for you. Supports gfx908/90a/942/950 (CDNA)
and gfx1100/1201 (RDNA). The recommended path over raw `__builtin_amdgcn_mfma_*`
when you want portable MMA. MIT.

## Reference

- Upstream: <https://github.com/ROCm/rocWMMA>
