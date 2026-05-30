---
id: pr-flash-attention-7
repo: ROCm/flash-attention
pr: 7
title: Increasing the compiling time by spliting into several cpp files
author: dejay-vu
date: '2023-07-31'
url: https://github.com/ROCm/flash-attention/pull/7
source_category: upstream-code
architectures:
- gfx90a
tags:
- bf16
- flash-attention
- fp16
- gemm
- gfx90a
techniques: []
hardware_features:
- bf16
- fp16
kernel_types:
- flash-attention
- gemm
languages:
- composable-kernel
- triton
- hip
captured_at: '2026-05-15'
status: merged
merge_sha: 0821eb0287f2
inclusion_reason: kernel path 'csrc/flash_attn_rocm/CMakeLists.txt'; keyword 'gemm'
changed_paths:
- .gitignore
- csrc/flash_attn_rocm/CMakeLists.txt
- csrc/flash_attn_rocm/composable_kernel
- csrc/flash_attn_rocm/fmha_api.cpp
- csrc/flash_attn_rocm/src/bwd_device_gemm_launcher.h
- csrc/flash_attn_rocm/src/bwd_device_gemm_template.h
- csrc/flash_attn_rocm/src/device_gemm_trait.h
- csrc/flash_attn_rocm/src/flash_bwd_runner_gfx90a.h
- csrc/flash_attn_rocm/src/flash_bwd_runner_kloop_hdim128_bf16_causal_gfx90a.cpp
- csrc/flash_attn_rocm/src/flash_bwd_runner_kloop_hdim128_bf16_noncausal_gfx90a.cpp
- csrc/flash_attn_rocm/src/flash_bwd_runner_kloop_hdim128_fp16_causal_gfx90a.cpp
- csrc/flash_attn_rocm/src/flash_bwd_runner_kloop_hdim128_fp16_noncausal_gfx90a.cpp
- csrc/flash_attn_rocm/src/flash_bwd_runner_kloop_hdim32_bf16_causal_gfx90a.cpp
- csrc/flash_attn_rocm/src/flash_bwd_runner_kloop_hdim32_bf16_noncausal_gfx90a.cpp
- csrc/flash_attn_rocm/src/flash_bwd_runner_kloop_hdim32_fp16_causal_gfx90a.cpp
- csrc/flash_attn_rocm/src/flash_bwd_runner_kloop_hdim32_fp16_noncausal_gfx90a.cpp
- csrc/flash_attn_rocm/src/flash_bwd_runner_kloop_hdim64_bf16_causal_gfx90a.cpp
- csrc/flash_attn_rocm/src/flash_bwd_runner_kloop_hdim64_bf16_noncausal_gfx90a.cpp
- csrc/flash_attn_rocm/src/flash_bwd_runner_kloop_hdim64_fp16_causal_gfx90a.cpp
- csrc/flash_attn_rocm/src/flash_bwd_runner_kloop_hdim64_fp16_noncausal_gfx90a.cpp
---

# Increasing the compiling time by spliting into several cpp files

**Repository:** [ROCm/flash-attention](https://github.com/ROCm/flash-attention) · **PR:** [#7](https://github.com/ROCm/flash-attention/pull/7) · **Merged:** 2023-07-31 · **Author:** @dejay-vu

**Inclusion reason:** kernel path 'csrc/flash_attn_rocm/CMakeLists.txt'; keyword 'gemm'

## Summary (from upstream PR description)

This is a tentative PR which has issues on PyTorch 1.13.1 so it is still under development.
Tested the elapsed time of "python setup.py install" on ROCm5.7/PyTorch 1.13.1:
Older version: 26m1.244s
This version: 4m11.111s on PyTorch 1.13.1
3m39.470s on PyTorch 2.0.1
Unit tests passed on ROCm5.7 + PyTorch 1.13.1: docker pull compute-artifactory.amd.com:5000/rocm-plus-docker/framework/compute-rocm-dkms-no-npi-hipclang:12505_ubuntu20.04_py3.8_pytorch_release-1.13_85fcc08
2113 passed, 2848 skipped in 119.70s

## Changed files (68 total, first 20 shown)

- `.gitignore`
- `csrc/flash_attn_rocm/CMakeLists.txt`
- `csrc/flash_attn_rocm/composable_kernel`
- `csrc/flash_attn_rocm/fmha_api.cpp`
- `csrc/flash_attn_rocm/src/bwd_device_gemm_launcher.h`
- `csrc/flash_attn_rocm/src/bwd_device_gemm_template.h`
- `csrc/flash_attn_rocm/src/device_gemm_trait.h`
- `csrc/flash_attn_rocm/src/flash_bwd_runner_gfx90a.h`
- `csrc/flash_attn_rocm/src/flash_bwd_runner_kloop_hdim128_bf16_causal_gfx90a.cpp`
- `csrc/flash_attn_rocm/src/flash_bwd_runner_kloop_hdim128_bf16_noncausal_gfx90a.cpp`
- `csrc/flash_attn_rocm/src/flash_bwd_runner_kloop_hdim128_fp16_causal_gfx90a.cpp`
- `csrc/flash_attn_rocm/src/flash_bwd_runner_kloop_hdim128_fp16_noncausal_gfx90a.cpp`
- `csrc/flash_attn_rocm/src/flash_bwd_runner_kloop_hdim32_bf16_causal_gfx90a.cpp`
- `csrc/flash_attn_rocm/src/flash_bwd_runner_kloop_hdim32_bf16_noncausal_gfx90a.cpp`
- `csrc/flash_attn_rocm/src/flash_bwd_runner_kloop_hdim32_fp16_causal_gfx90a.cpp`
- `csrc/flash_attn_rocm/src/flash_bwd_runner_kloop_hdim32_fp16_noncausal_gfx90a.cpp`
- `csrc/flash_attn_rocm/src/flash_bwd_runner_kloop_hdim64_bf16_causal_gfx90a.cpp`
- `csrc/flash_attn_rocm/src/flash_bwd_runner_kloop_hdim64_bf16_noncausal_gfx90a.cpp`
- `csrc/flash_attn_rocm/src/flash_bwd_runner_kloop_hdim64_fp16_causal_gfx90a.cpp`
- `csrc/flash_attn_rocm/src/flash_bwd_runner_kloop_hdim64_fp16_noncausal_gfx90a.cpp`

## Provenance

- Merge commit: `0821eb0287f2`
- Captured at knowledge cutoff: 2026-05-15
- Source of truth: the upstream PR linked above. This page summarizes upstream
  metadata; consult the PR for the authoritative diff.
