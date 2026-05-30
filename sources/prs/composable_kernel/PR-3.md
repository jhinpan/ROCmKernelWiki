---
id: pr-composable_kernel-3
repo: ROCm/composable_kernel
pr: 3
title: Update to clang-format-10
author: asroy
date: '2021-07-30'
url: https://github.com/ROCm/composable_kernel/pull/3
source_category: upstream-code
architectures:
- gfx942
tags:
- fp16
- gemm
- gfx942
techniques: []
hardware_features:
- fp16
kernel_types:
- gemm
languages:
- composable-kernel
- hip
- cpp
captured_at: '2026-05-15'
status: merged
merge_sha: f6edda6119eb
inclusion_reason: kernel path 'composable_kernel/include/tensor_description/dynamic_tensor_descriptor.hpp';
  keyword 'gemm'
changed_paths:
- composable_kernel/include/tensor_description/dynamic_tensor_descriptor.hpp
- composable_kernel/include/tensor_description/tensor_adaptor.hpp
- composable_kernel/include/tensor_operation/blockwise_gemm_dlops_v3.hpp
- composable_kernel/include/tensor_operation/threadwise_contraction_dlops.hpp
- composable_kernel/include/tensor_operation/threadwise_gemm_dlops_v3.hpp
- composable_kernel/include/utility/data_type_enum.hpp
- external/half/include/half.hpp
- host/host_tensor/include/host_tensor.hpp
- host/online_compilation/hip_utility/kernel_cache.cpp
- host/online_compilation/hip_utility/logger.cpp
---

# Update to clang-format-10

**Repository:** [ROCm/composable_kernel](https://github.com/ROCm/composable_kernel) · **PR:** [#3](https://github.com/ROCm/composable_kernel/pull/3) · **Merged:** 2021-07-30 · **Author:** @asroy

**Inclusion reason:** kernel path 'composable_kernel/include/tensor_description/dynamic_tensor_descriptor.hpp'; keyword 'gemm'

## Summary (from upstream PR description)

_No PR description provided upstream._

## Changed files (10 total, first 10 shown)

- `composable_kernel/include/tensor_description/dynamic_tensor_descriptor.hpp`
- `composable_kernel/include/tensor_description/tensor_adaptor.hpp`
- `composable_kernel/include/tensor_operation/blockwise_gemm_dlops_v3.hpp`
- `composable_kernel/include/tensor_operation/threadwise_contraction_dlops.hpp`
- `composable_kernel/include/tensor_operation/threadwise_gemm_dlops_v3.hpp`
- `composable_kernel/include/utility/data_type_enum.hpp`
- `external/half/include/half.hpp`
- `host/host_tensor/include/host_tensor.hpp`
- `host/online_compilation/hip_utility/kernel_cache.cpp`
- `host/online_compilation/hip_utility/logger.cpp`

## Provenance

- Merge commit: `f6edda6119eb`
- Captured at knowledge cutoff: 2026-05-15
- Source of truth: the upstream PR linked above. This page summarizes upstream
  metadata; consult the PR for the authoritative diff.
