---
id: ref-flydsl
title: FlyDSL — Flexible Layout DSL for AMD GPUs
repo: ROCm/FlyDSL
url: https://github.com/ROCm/FlyDSL
author: ROCm
source_category: reference-repo
architectures:
- gfx942
- gfx950
- gfx1201
tags:
- flydsl
- mfma
- gemm
- preshuffle-layout
languages:
- flydsl
- mlir
- python
retrieved_at: '2026-05-15'
---

# FlyDSL — Flexible Layout DSL for AMD GPUs

FlyDSL is a Python DSL plus an MLIR-native compiler stack for
authoring AMD GPU kernels with explicit layouts and tiling. The `fly` dialect is
a layout IR (`!fly.int_tuple`, `!fly.layout`, `!fly.coord_tensor`, `!fly.memref`)
with CuTe-like layout algebra: a Layout is a (Shape, Stride) pair mapping a
coordinate to a linear index. Kernels use `@flyc.kernel` / `@flyc.jit`; the JIT
traces Python to MLIR and lowers Fly → ROCDL → LLVM → fatbin. Tiling is explicit
across block/warp/thread/instruction scopes with MFMA atoms. Verified targets:
MI300X/MI308X (gfx942), MI350/MI355X (gfx950), MI450 (gfx1250), Radeon AI PRO
R9700 (gfx1201). Apache-2.0. Examples: `01-vectorAdd.py`, `02-tiledCopy.py`,
`03-tiledMma.py`, `04-preshuffle_gemm.py`. Note: experimental, not part of the
official ROCm distribution.

## Reference

- Upstream: <https://github.com/ROCm/FlyDSL>
