---
id: pr-composable_kernel-4
repo: ROCm/composable_kernel
pr: 4
title: Separate online compile
author: asroy
date: '2021-08-06'
url: https://github.com/ROCm/composable_kernel/pull/4
source_category: upstream-code
architectures:
- gfx942
tags:
- convolution
- fp16
- gemm
- gfx942
- mfma
- rope
techniques: []
hardware_features:
- fp16
- mfma
kernel_types:
- convolution
- gemm
- rope
languages:
- composable-kernel
- hip
- cpp
captured_at: '2026-05-15'
artifact_dir: artifacts/prs/composable_kernel/PR-4
status: merged
merge_sha: 97e6d514f7b2
inclusion_reason: kernel path 'host/driver_online/include/online_device_dynamic_convolution_forward_implicit_gemm_v4r4_dlops_nchw_kcyx_nkhw.hpp';
  keyword 'gemm'
changed_paths:
- external/half/include/half.hpp
- host/CMakeLists.txt
- host/driver_offline/CMakeLists.txt
- host/driver_online/CMakeLists.txt
- host/driver_online/conv_fwd_driver_online.cpp
- host/driver_online/include/online_device_dynamic_convolution_forward_implicit_gemm_v4r4_dlops_nchw_kcyx_nkhw.hpp
- host/driver_online/include/online_device_dynamic_convolution_forward_implicit_gemm_v4r4_xdlops_nchw_kcyx_nkhw.hpp
- host/driver_online/include/online_device_dynamic_convolution_forward_implicit_gemm_v4r4_xdlops_nhwc_kyxc_nhwk.hpp
- host/driver_online/include/online_device_dynamic_convolution_forward_implicit_gemm_v6r1_dlops_nchw_kcyx_nkhw.hpp
- host/driver_online/include/online_driver_common.hpp
- host/online_compile/CMakeLists.txt
- host/online_compile/addkernels/CMakeLists.txt
- host/online_compile/addkernels/addkernels.cpp
- host/online_compile/addkernels/include_inliner.cpp
- host/online_compile/addkernels/include_inliner.hpp
- host/online_compile/addkernels/source_file_desc.hpp
- host/online_compile/hip_utility/binary_cache.cpp
- host/online_compile/hip_utility/exec_utils.cpp
- host/online_compile/hip_utility/handlehip.cpp
- host/online_compile/hip_utility/hip_build_utils.cpp
facet_source: inferred
related:
- technique-split-k
- technique-preshuffle-layout
- technique-mfma-pipelining
---
# Separate online compile

**Repository:** [ROCm/composable_kernel](https://github.com/ROCm/composable_kernel) · **PR:** [#4](https://github.com/ROCm/composable_kernel/pull/4) · **Merged:** 2021-08-06 · **Author:** @asroy

**Inclusion reason:** kernel path 'host/driver_online/include/online_device_dynamic_convolution_forward_implicit_gemm_v4r4_dlops_nchw_kcyx_nkhw.hpp'; keyword 'gemm'

## Summary (from upstream PR description)

_No PR description provided upstream._

## Changed files (58 total, first 20 shown)

- `external/half/include/half.hpp`
- `host/CMakeLists.txt`
- `host/driver_offline/CMakeLists.txt`
- `host/driver_online/CMakeLists.txt`
- `host/driver_online/conv_fwd_driver_online.cpp`
- `host/driver_online/include/online_device_dynamic_convolution_forward_implicit_gemm_v4r4_dlops_nchw_kcyx_nkhw.hpp`
- `host/driver_online/include/online_device_dynamic_convolution_forward_implicit_gemm_v4r4_xdlops_nchw_kcyx_nkhw.hpp`
- `host/driver_online/include/online_device_dynamic_convolution_forward_implicit_gemm_v4r4_xdlops_nhwc_kyxc_nhwk.hpp`
- `host/driver_online/include/online_device_dynamic_convolution_forward_implicit_gemm_v6r1_dlops_nchw_kcyx_nkhw.hpp`
- `host/driver_online/include/online_driver_common.hpp`
- `host/online_compile/CMakeLists.txt`
- `host/online_compile/addkernels/CMakeLists.txt`
- `host/online_compile/addkernels/addkernels.cpp`
- `host/online_compile/addkernels/include_inliner.cpp`
- `host/online_compile/addkernels/include_inliner.hpp`
- `host/online_compile/addkernels/source_file_desc.hpp`
- `host/online_compile/hip_utility/binary_cache.cpp`
- `host/online_compile/hip_utility/exec_utils.cpp`
- `host/online_compile/hip_utility/handlehip.cpp`
- `host/online_compile/hip_utility/hip_build_utils.cpp`

## Provenance

- Merge commit: `97e6d514f7b2`
- Captured at knowledge cutoff: 2026-05-15
- Source of truth: the upstream PR linked above. This page summarizes upstream
  metadata; consult the PR for the authoritative diff.
