---
id: pr-flash-attention-2
repo: ROCm/flash-attention
pr: 2
title: Jhzhan/release test
author: sabreshao
date: '2023-06-02'
url: https://github.com/ROCm/flash-attention/pull/2
source_category: upstream-code
architectures:
- gfx90a
tags:
- attention
- bf16
- flash-attention
- fp16
- fp8
- gfx90a
techniques: []
hardware_features:
- bf16
- fp16
- fp8
kernel_types:
- attention
- flash-attention
languages:
- composable-kernel
- triton
- hip
captured_at: '2026-05-15'
status: merged
merge_sha: 782e7abf88ac
inclusion_reason: kernel path 'csrc/flash_attn_rocm/composable_kernel'; keyword 'flash'
changed_paths:
- .gitignore
- Dockerfile.rocm
- README.md
- csrc/flash_attn_rocm/composable_kernel
- csrc/flash_attn_rocm/fmha_api.cpp
- csrc/flash_attn_rocm/src/fmha.h
- csrc/flash_attn_rocm/src/fmha_dgrad_fp16_bf16_kernel.gfx90a.cpp
- csrc/flash_attn_rocm/src/fmha_fprop_fp16_bf16_kernel.gfx90a.cpp
- csrc/flash_attn_rocm/src/fmha_utils.h
- flash_attn/flash_attn_interface.py
- hipify_patch.patch
- setup.py
- tests/test_flash_attn.py
facet_source: inferred
related:
- kernel-flash-attention-ck
- kernel-mla-decode
- technique-vgpr-budgeting
---
# Jhzhan/release test

**Repository:** [ROCm/flash-attention](https://github.com/ROCm/flash-attention) · **PR:** [#2](https://github.com/ROCm/flash-attention/pull/2) · **Merged:** 2023-06-02 · **Author:** @sabreshao

**Inclusion reason:** kernel path 'csrc/flash_attn_rocm/composable_kernel'; keyword 'flash'

## Summary (from upstream PR description)

_No PR description provided upstream._

## Changed files (13 total, first 13 shown)

- `.gitignore`
- `Dockerfile.rocm`
- `README.md`
- `csrc/flash_attn_rocm/composable_kernel`
- `csrc/flash_attn_rocm/fmha_api.cpp`
- `csrc/flash_attn_rocm/src/fmha.h`
- `csrc/flash_attn_rocm/src/fmha_dgrad_fp16_bf16_kernel.gfx90a.cpp`
- `csrc/flash_attn_rocm/src/fmha_fprop_fp16_bf16_kernel.gfx90a.cpp`
- `csrc/flash_attn_rocm/src/fmha_utils.h`
- `flash_attn/flash_attn_interface.py`
- `hipify_patch.patch`
- `setup.py`
- `tests/test_flash_attn.py`

## Provenance

- Merge commit: `782e7abf88ac`
- Captured at knowledge cutoff: 2026-05-15
- Source of truth: the upstream PR linked above. This page summarizes upstream
  metadata; consult the PR for the authoritative diff.
