---
id: ref-hipblaslt
title: hipBLASLt — GEMM with epilogue fusion
repo: ROCm/hipBLASLt
url: https://github.com/ROCm/hipBLASLt
author: ROCm
source_category: reference-repo
architectures:
- gfx942
- gfx950
tags:
- gemm
- fp8-gemm
- epilogue-fusion
languages:
- hip
- cpp
- gcn-asm
retrieved_at: '2026-05-15'
---

# hipBLASLt — GEMM with epilogue fusion

hipBLASLt is AMD's lightweight GEMM library (cuBLASLt-style API,
`hipblasLtMatmul`) computing D = Activation(alpha·op(A)·op(B) + beta·op(C) + bias)
with GELU/ReLU/Swish epilogues and bias fusion. FP8 support distinguishes FNUZ
(gfx942) from OCP (gfx950) types. Its kernel generator backend is TensileLite,
which emits AMDGPU assembly GEMM kernels selected per problem size/dtype.

## Reference

- Upstream: <https://github.com/ROCm/hipBLASLt>
