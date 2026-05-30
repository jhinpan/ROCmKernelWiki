---
id: pr-aiter-5
repo: ROCm/aiter
pr: 5
title: Transpose add
author: fangche123
date: '2024-12-04'
url: https://github.com/ROCm/aiter/pull/5
source_category: upstream-code
architectures:
- gfx942
tags:
- gfx942
- moe
techniques: []
hardware_features: []
kernel_types:
- moe
languages:
- hip
- cpp
- triton
captured_at: '2026-05-15'
status: merged
merge_sha: 8db72fcb206b
inclusion_reason: kernel path 'csrc/moe_ops.h'; keyword 'moe'
changed_paths:
- ater/__init__.py
- ater/transpose_operator.py
- csrc/moe_ops.h
- csrc/rocm_ops.cpp
- csrc/transpose_add.cu
- csrc/transpose_operator.cu
- op_tests/dense_70x2.py
- op_tests/test_transpose_add.py
---

# Transpose add

**Repository:** [ROCm/aiter](https://github.com/ROCm/aiter) · **PR:** [#5](https://github.com/ROCm/aiter/pull/5) · **Merged:** 2024-12-04 · **Author:** @fangche123

**Inclusion reason:** kernel path 'csrc/moe_ops.h'; keyword 'moe'

## Summary (from upstream PR description)

_No PR description provided upstream._

## Changed files (8 total, first 8 shown)

- `ater/__init__.py`
- `ater/transpose_operator.py`
- `csrc/moe_ops.h`
- `csrc/rocm_ops.cpp`
- `csrc/transpose_add.cu`
- `csrc/transpose_operator.cu`
- `op_tests/dense_70x2.py`
- `op_tests/test_transpose_add.py`

## Provenance

- Merge commit: `8db72fcb206b`
- Captured at knowledge cutoff: 2026-05-15
- Source of truth: the upstream PR linked above. This page summarizes upstream
  metadata; consult the PR for the authoritative diff.
