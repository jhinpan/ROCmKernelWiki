---
id: pr-composable_kernel-8
repo: ROCm/composable_kernel
pr: 8
title: 'MIOpen Downstream: Initial integration'
author: asroy
date: '2021-08-16'
url: https://github.com/ROCm/composable_kernel/pull/8
source_category: upstream-code
architectures:
- gfx942
tags:
- gemm
- gfx942
techniques: []
hardware_features: []
kernel_types:
- gemm
languages:
- composable-kernel
- hip
- cpp
captured_at: '2026-05-15'
status: merged
merge_sha: ccc4a1d36599
inclusion_reason: kernel path 'cmake/AddKernels.cmake'; keyword 'gemm'
changed_paths:
- .clang-tidy
- CMakeLists.txt
- README.md
- cmake/AddKernels.cmake
- cmake/Analyzers.cmake
- cmake/ClangTidy.cmake
- cmake/CppCheck.cmake
- cmake/DoxygenDoc.cmake
- cmake/EnableCompilerWarnings.cmake
- cmake/TargetFlags.cmake
- composable_kernel/include/problem_transform/transform_backward_data_convolution_into_gemm_v4r1_nhwc_kyxc_nhwk.hpp
- composable_kernel/include/problem_transform/transform_backward_data_convolution_into_gemm_v4r1r2_nhwc_kyxc_nhwk.hpp
- composable_kernel/include/problem_transform/transform_forward_convolution_into_gemm_v4r4_nchw_kcyx_nkhw.hpp
- composable_kernel/include/problem_transform/transform_forward_convolution_into_gemm_v4r4_nhwc_kyxc_nhwk.hpp
- composable_kernel/include/problem_transform/transform_forward_convolution_into_gemm_v4r4r2_nchw_kcyx_nkhw.hpp
- composable_kernel/include/problem_transform/transform_forward_convolution_into_gemm_v4r4r2_nhwc_kyxc_nhwk.hpp
- composable_kernel/include/problem_transform/transform_forward_convolution_into_gemm_v4r4r4_nhwc_kyxc_nhwk.hpp
- composable_kernel/include/problem_transform/transform_forward_convolution_into_gemm_v6r1_nchw_kcyx_nkhw.hpp
- composable_kernel/include/tensor_description/multi_index_transform.hpp
- composable_kernel/include/tensor_description/multi_index_transform_helper.hpp
---

# MIOpen Downstream: Initial integration

**Repository:** [ROCm/composable_kernel](https://github.com/ROCm/composable_kernel) · **PR:** [#8](https://github.com/ROCm/composable_kernel/pull/8) · **Merged:** 2021-08-16 · **Author:** @asroy

**Inclusion reason:** kernel path 'cmake/AddKernels.cmake'; keyword 'gemm'

## Summary (from upstream PR description)

Downstream changes from MIOpen
ROCm/MIOpen#1071

## Changed files (150 total, first 20 shown)

- `.clang-tidy`
- `CMakeLists.txt`
- `README.md`
- `cmake/AddKernels.cmake`
- `cmake/Analyzers.cmake`
- `cmake/ClangTidy.cmake`
- `cmake/CppCheck.cmake`
- `cmake/DoxygenDoc.cmake`
- `cmake/EnableCompilerWarnings.cmake`
- `cmake/TargetFlags.cmake`
- `composable_kernel/include/problem_transform/transform_backward_data_convolution_into_gemm_v4r1_nhwc_kyxc_nhwk.hpp`
- `composable_kernel/include/problem_transform/transform_backward_data_convolution_into_gemm_v4r1r2_nhwc_kyxc_nhwk.hpp`
- `composable_kernel/include/problem_transform/transform_forward_convolution_into_gemm_v4r4_nchw_kcyx_nkhw.hpp`
- `composable_kernel/include/problem_transform/transform_forward_convolution_into_gemm_v4r4_nhwc_kyxc_nhwk.hpp`
- `composable_kernel/include/problem_transform/transform_forward_convolution_into_gemm_v4r4r2_nchw_kcyx_nkhw.hpp`
- `composable_kernel/include/problem_transform/transform_forward_convolution_into_gemm_v4r4r2_nhwc_kyxc_nhwk.hpp`
- `composable_kernel/include/problem_transform/transform_forward_convolution_into_gemm_v4r4r4_nhwc_kyxc_nhwk.hpp`
- `composable_kernel/include/problem_transform/transform_forward_convolution_into_gemm_v6r1_nchw_kcyx_nkhw.hpp`
- `composable_kernel/include/tensor_description/multi_index_transform.hpp`
- `composable_kernel/include/tensor_description/multi_index_transform_helper.hpp`

## Provenance

- Merge commit: `ccc4a1d36599`
- Captured at knowledge cutoff: 2026-05-15
- Source of truth: the upstream PR linked above. This page summarizes upstream
  metadata; consult the PR for the authoritative diff.
