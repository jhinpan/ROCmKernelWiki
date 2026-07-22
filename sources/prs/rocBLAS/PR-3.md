---
id: pr-rocBLAS-3
repo: ROCm/rocBLAS
pr: 3
title: 'first major ablas update: add netlib/batched blas api. elaborate ablas types'
author: tingxingdong
date: '2016-02-09'
url: https://github.com/ROCm/rocBLAS/pull/3
source_category: upstream-code
architectures:
- gfx942
tags:
- gemm
- gfx942
- moe
- sgemm
techniques: []
hardware_features: []
kernel_types:
- gemm
- moe
- sgemm
languages:
- hip
- cpp
captured_at: '2026-05-15'
status: merged
merge_sha: 774f0bd62c3233de81d5b220f552ab19b773cc06
inclusion_reason: keyword 'gemm'
changed_paths:
- src/include/ablas.h
- src/include/ablas_expert.h
- src/include/ablas_flops.h
- src/include/ablas_netlib.h
- src/include/ablas_netlib_batched.h
- src/include/ablas_runtime.h
- src/include/ablas_types.h
- src/include/ablas_utility.h
facet_source: inferred
related:
- kernel-grouped-gemm
- technique-vgpr-budgeting
- technique-stream-k
---
# first major ablas update: add netlib/batched blas api. elaborate ablas types

**Repository:** [ROCm/rocBLAS](https://github.com/ROCm/rocBLAS) · **PR:** [#3](https://github.com/ROCm/rocBLAS/pull/3) · **Merged:** 2016-02-09 · **Author:** @tingxingdong

**Inclusion reason:** keyword 'gemm'

## Summary (from upstream PR description)

Define various ablas types. Honor current main-streamed BLAS implementation, and use integer to specify matrix options, like "ablas_notrans   = 111" (see ablas_types.h).


As we agreed, and also follow the existing naming style already committed in repo.
The name is like ablas_notrans, instead of ablasNotrans.
so, it is ablas_sgemm instead of ablasSgemm


Add Netlib and batched BLAS routines interface. Currently, the return type is void. Yet, it should be easy to "find and replace" after we define a return type like, ablas_status.


In order to support int64 in the future, we use "ablas_int" everywhere rather than int. But like other BLAS library, we do NOT introduce artificial "ablas_float" to wrapper float. But we do have ablas_floatcomplex.

## Changed files (8 total, first 8 shown)

- `src/include/ablas.h`
- `src/include/ablas_expert.h`
- `src/include/ablas_flops.h`
- `src/include/ablas_netlib.h`
- `src/include/ablas_netlib_batched.h`
- `src/include/ablas_runtime.h`
- `src/include/ablas_types.h`
- `src/include/ablas_utility.h`

## Provenance

- Merge commit: ``
- Captured at knowledge cutoff: 2026-05-15
- Source of truth: the upstream PR linked above. This page summarizes upstream
  metadata; consult the PR for the authoritative diff.
