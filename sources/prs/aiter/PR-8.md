---
id: pr-aiter-8
repo: ROCm/aiter
pr: 8
title: fix err
author: carlushuang
date: '2024-12-11'
url: https://github.com/ROCm/aiter/pull/8
source_category: upstream-code
architectures:
- gfx942
tags:
- attention
- gfx942
techniques: []
hardware_features: []
kernel_types:
- attention
languages:
- hip
- cpp
- triton
captured_at: '2026-05-15'
status: merged
merge_sha: 4e47eb97f18f
inclusion_reason: kernel path 'csrc/py_itfs_ck/attention_kernels.cu'; keyword 'attention'
changed_paths:
- csrc/py_itfs_ck/attention_kernels.cu
- op_tests/test_pa.py
related:
- lang-flydsl
- kernel-paged-attention
- kernel-mla-decode
---
# fix err

**Repository:** [ROCm/aiter](https://github.com/ROCm/aiter) · **PR:** [#8](https://github.com/ROCm/aiter/pull/8) · **Merged:** 2024-12-11 · **Author:** @carlushuang

**Inclusion reason:** kernel path 'csrc/py_itfs_ck/attention_kernels.cu'; keyword 'attention'

## Summary (from upstream PR description)

_No PR description provided upstream._

## Changed files (2 total, first 2 shown)

- `csrc/py_itfs_ck/attention_kernels.cu`
- `op_tests/test_pa.py`

## Provenance

- Merge commit: `4e47eb97f18f`
- Captured at knowledge cutoff: 2026-05-15
- Source of truth: the upstream PR linked above. This page summarizes upstream
  metadata; consult the PR for the authoritative diff.
