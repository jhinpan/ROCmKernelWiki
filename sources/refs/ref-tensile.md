---
id: ref-tensile
title: Tensile — assembly GEMM kernel generator
repo: ROCm/Tensile
url: https://github.com/ROCm/Tensile
author: ROCm
source_category: reference-repo
architectures:
- gfx942
- gfx950
tags:
- gemm
- gcn-asm
- tile-scheduling
- split-k
languages:
- gcn-asm
- python
retrieved_at: '2026-05-15'
---

# Tensile — assembly GEMM kernel generator

Tensile is a Python tool that generates benchmark-driven GEMM (and
tensor-contraction) backend libraries, mainly for rocBLAS. Its `KernelLanguage`
parameter chooses HIP or assembly output; the `MatrixInstruction` parameter
encodes the MFMA shape and wave-tiling ([M,N,K,B, WaveTileM/N, WaveGroupM/N]).
Solution selection runs a four-level catalog (hardware → operation → problem →
exact solution) with performance-ranked kernels.

## Reference

- Upstream: <https://github.com/ROCm/Tensile>
