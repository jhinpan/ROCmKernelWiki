---
id: pr-rocBLAS-8
repo: ROCm/rocBLAS
pr: 8
title: cmake build infrastructure for library clients
author: kknox
date: '2016-03-02'
url: https://github.com/ROCm/rocBLAS/pull/8
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
- gemm
- moe
languages:
- hip
- cpp
captured_at: '2026-05-15'
status: merged
merge_sha: ''
inclusion_reason: kernel path 'clients/samples/gemm-expert/CMakeLists.txt'; keyword
  'gemm'
changed_paths:
- .gitignore
- CMakeLists.txt
- clients/CMakeLists.txt
- clients/cmake/ExternalGmock.cmake
- clients/cmake/build-options.cmake
- clients/cmake/external-boost.cmake
- clients/cmake/external-gtest.cmake
- clients/samples/CMakeLists.txt
- clients/samples/gemm-expert/CMakeLists.txt
- clients/samples/gemm-expert/gemm-expert.cpp
- src/CMakeLists.txt
- src/library/CMakeLists.txt
facet_source: inferred
related:
- kernel-grouped-gemm
- technique-vgpr-budgeting
- technique-stream-k
---
# cmake build infrastructure for library clients

**Repository:** [ROCm/rocBLAS](https://github.com/ROCm/rocBLAS) · **PR:** [#8](https://github.com/ROCm/rocBLAS/pull/8) · **Merged:** 2016-03-02 · **Author:** @kknox

**Inclusion reason:** kernel path 'clients/samples/gemm-expert/CMakeLists.txt'; keyword 'gemm'

## Summary (from upstream PR description)

This has enough functionality to create a small executable sample which links with the main library.  Also, boost and googletest dependencies are built.

## Changed files (12 total, first 12 shown)

- `.gitignore`
- `CMakeLists.txt`
- `clients/CMakeLists.txt`
- `clients/cmake/ExternalGmock.cmake`
- `clients/cmake/build-options.cmake`
- `clients/cmake/external-boost.cmake`
- `clients/cmake/external-gtest.cmake`
- `clients/samples/CMakeLists.txt`
- `clients/samples/gemm-expert/CMakeLists.txt`
- `clients/samples/gemm-expert/gemm-expert.cpp`
- `src/CMakeLists.txt`
- `src/library/CMakeLists.txt`

## Provenance

- Merge commit: ``
- Captured at knowledge cutoff: 2026-05-15
- Source of truth: the upstream PR linked above. This page summarizes upstream
  metadata; consult the PR for the authoritative diff.
