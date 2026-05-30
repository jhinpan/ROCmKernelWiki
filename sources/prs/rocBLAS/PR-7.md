---
id: pr-rocBLAS-7
repo: ROCm/rocBLAS
pr: 7
title: superbuild & cmake infrastructure for the library
author: kknox
date: '2016-02-26'
url: https://github.com/ROCm/rocBLAS/pull/7
source_category: upstream-code
architectures:
- gfx942
tags:
- gemm
- gfx942
- moe
techniques: []
hardware_features: []
kernel_types:
- moe
- gemm
languages:
- hip
- cpp
captured_at: '2026-05-15'
status: merged
merge_sha: ''
inclusion_reason: kernel path 'src/library/blas3/gemm.cpp'; keyword 'gemm'
changed_paths:
- .github/CONTRIBUTING.md
- CMakeLists.txt
- Questions.md
- README.md
- clients/CMakeLists.txt
- clients/benchmarks/CMakeLists.txt
- clients/cmake/build-options.cmake
- cmake/CMakeLists.txt
- cmake/clang-toolchain.cmake
- cmake/hcc-toolchain.cmake
- src/CMakeLists.txt
- src/cmake/CMakeLists.txt
- src/cmake/build-options.cmake
- src/include/ablas.h
- src/include/ablas_types.h
- src/include/rocblas-expert.h
- src/include/rocblas-types.h
- src/include/rocblas-version.h.in
- src/include/rocblas.h
- src/library/CMakeLists.txt
---

# superbuild & cmake infrastructure for the library

**Repository:** [ROCm/rocBLAS](https://github.com/ROCm/rocBLAS) · **PR:** [#7](https://github.com/ROCm/rocBLAS/pull/7) · **Merged:** 2016-02-26 · **Author:** @kknox

**Inclusion reason:** kernel path 'src/library/blas3/gemm.cpp'; keyword 'gemm'

## Summary (from upstream PR description)

_No PR description provided upstream._

## Changed files (23 total, first 20 shown)

- `.github/CONTRIBUTING.md`
- `CMakeLists.txt`
- `Questions.md`
- `README.md`
- `clients/CMakeLists.txt`
- `clients/benchmarks/CMakeLists.txt`
- `clients/cmake/build-options.cmake`
- `cmake/CMakeLists.txt`
- `cmake/clang-toolchain.cmake`
- `cmake/hcc-toolchain.cmake`
- `src/CMakeLists.txt`
- `src/cmake/CMakeLists.txt`
- `src/cmake/build-options.cmake`
- `src/include/ablas.h`
- `src/include/ablas_types.h`
- `src/include/rocblas-expert.h`
- `src/include/rocblas-types.h`
- `src/include/rocblas-version.h.in`
- `src/include/rocblas.h`
- `src/library/CMakeLists.txt`

## Provenance

- Merge commit: ``
- Captured at knowledge cutoff: 2026-05-15
- Source of truth: the upstream PR linked above. This page summarizes upstream
  metadata; consult the PR for the authoritative diff.
