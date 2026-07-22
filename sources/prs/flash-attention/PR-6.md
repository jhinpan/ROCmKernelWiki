---
id: pr-flash-attention-6
repo: ROCm/flash-attention
pr: 6
title: Enable both Qloop and Kloop
author: guangzlu
date: '2023-07-14'
url: https://github.com/ROCm/flash-attention/pull/6
source_category: upstream-code
architectures:
- gfx90a
tags:
- attention
- bf16
- flash-attention
- fp16
- gfx90a
techniques: []
hardware_features:
- bf16
- fp16
kernel_types:
- attention
- flash-attention
languages:
- composable-kernel
- triton
- hip
captured_at: '2026-05-15'
status: merged
merge_sha: 95514492a064
inclusion_reason: kernel path 'csrc/flash_attn_rocm/composable_kernel'; keyword 'attention'
changed_paths:
- README.md
- csrc/flash_attn_rocm/composable_kernel
- csrc/flash_attn_rocm/fmha_api.cpp
- csrc/flash_attn_rocm/src/fmha.h
- csrc/flash_attn_rocm/src/fmha_dgrad_fp16_bf16_kernel.gfx90a.cpp
- csrc/flash_attn_rocm/src/fmha_fprop_fp16_bf16_kernel.gfx90a.cpp
- csrc/flash_attn_rocm/src/fmha_utils.h
- flash_attn/flash_attn_interface.py
facet_source: inferred
---
# Enable both Qloop and Kloop

**Repository:** [ROCm/flash-attention](https://github.com/ROCm/flash-attention) · **PR:** [#6](https://github.com/ROCm/flash-attention/pull/6) · **Merged:** 2023-07-14 · **Author:** @guangzlu

**Inclusion reason:** kernel path 'csrc/flash_attn_rocm/composable_kernel'; keyword 'attention'

## Summary (from upstream PR description)

In this PR, both Qloop and Kloop are enabled.
By default we are using qloop. If you want to use kloop, please turn on the kloop flag by setting the environment variable:
export FLASH_ATTENTION_INTERNAL_USE_KLOOP=1
Here is a table of performance comparision between Qloop and Kloop.
kloop.vs.qloop.xlsx
(In this table, RTZ is used and we choosed function ' flash_attn_unpadded_func ' for test)
From the table, we can find that when comparing total performance (fwd + bwd), qloop is better in most cases. But when comparing fwd with dropout only, kloop is better. So you can choose the better route by your needs.

## Changed files (8 total, first 8 shown)

- `README.md`
- `csrc/flash_attn_rocm/composable_kernel`
- `csrc/flash_attn_rocm/fmha_api.cpp`
- `csrc/flash_attn_rocm/src/fmha.h`
- `csrc/flash_attn_rocm/src/fmha_dgrad_fp16_bf16_kernel.gfx90a.cpp`
- `csrc/flash_attn_rocm/src/fmha_fprop_fp16_bf16_kernel.gfx90a.cpp`
- `csrc/flash_attn_rocm/src/fmha_utils.h`
- `flash_attn/flash_attn_interface.py`

## Provenance

- Merge commit: `95514492a064`
- Captured at knowledge cutoff: 2026-05-15
- Source of truth: the upstream PR linked above. This page summarizes upstream
  metadata; consult the PR for the authoritative diff.
