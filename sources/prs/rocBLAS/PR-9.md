---
id: pr-rocBLAS-9
repo: ROCm/rocBLAS
pr: 9
title: Add hip dep
author: kknox
date: '2016-03-07'
url: https://github.com/ROCm/rocBLAS/pull/9
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
inclusion_reason: kernel path 'clients/samples/gemm-expert/gemm-expert.cpp'; keyword
  'gemm'
changed_paths:
- CMakeLists.txt
- clients/CMakeLists.txt
- clients/cmake/external-gtest.cmake
- clients/samples/CMakeLists.txt
- clients/samples/gemm-expert/gemm-expert.cpp
- cmake/clang-toolchain.cmake
- cmake/external-hip.cmake
- cmake/gcc-toolchain.cmake
- cmake/hipcc-toolchain.cmake.in
- src/CMakeLists.txt
- src/cmake/build-options.cmake
- src/library/blas3/gemm.cpp
---

# Add hip dep

**Repository:** [ROCm/rocBLAS](https://github.com/ROCm/rocBLAS) · **PR:** [#9](https://github.com/ROCm/rocBLAS/pull/9) · **Merged:** 2016-03-07 · **Author:** @kknox

**Inclusion reason:** kernel path 'clients/samples/gemm-expert/gemm-expert.cpp'; keyword 'gemm'

## Summary (from upstream PR description)

Adding ability for build infrastructure to download, build and link 'HiP' into library and client programs

## Changed files (12 total, first 12 shown)

- `CMakeLists.txt`
- `clients/CMakeLists.txt`
- `clients/cmake/external-gtest.cmake`
- `clients/samples/CMakeLists.txt`
- `clients/samples/gemm-expert/gemm-expert.cpp`
- `cmake/clang-toolchain.cmake`
- `cmake/external-hip.cmake`
- `cmake/gcc-toolchain.cmake`
- `cmake/hipcc-toolchain.cmake.in`
- `src/CMakeLists.txt`
- `src/cmake/build-options.cmake`
- `src/library/blas3/gemm.cpp`

## Provenance

- Merge commit: ``
- Captured at knowledge cutoff: 2026-05-15
- Source of truth: the upstream PR linked above. This page summarizes upstream
  metadata; consult the PR for the authoritative diff.
