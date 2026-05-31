---
id: pr-aiter-4
repo: ROCm/aiter
pr: 4
title: add add_transpose implement
author: fangche123
date: '2024-11-27'
url: https://github.com/ROCm/aiter/pull/4
source_category: upstream-code
architectures:
- gfx942
tags:
- gfx942
- moe
- transpose
techniques: []
hardware_features: []
kernel_types:
- moe
- transpose
languages:
- hip
- cpp
- triton
captured_at: '2026-05-15'
status: merged
merge_sha: 1a7637a4c440
inclusion_reason: kernel path 'csrc/moe_ops.h'; keyword 'moe'
changed_paths:
- csrc/moe_ops.h
- csrc/rocm_ops.cpp
- csrc/transpose_add.cu
- op_tests/dense_70x2.py
- op_tests/test_transpose_add.py
facet_source: inferred
related:
- technique-bank-conflict-avoidance
- kernel-transpose-lds
- kernel-grouped-gemm
---
# add add_transpose implement

**Repository:** [ROCm/aiter](https://github.com/ROCm/aiter) · **PR:** [#4](https://github.com/ROCm/aiter/pull/4) · **Merged:** 2024-11-27 · **Author:** @fangche123

**Inclusion reason:** kernel path 'csrc/moe_ops.h'; keyword 'moe'

## Summary (from upstream PR description)

_No PR description provided upstream._

## Changed files (5 total, first 5 shown)

- `csrc/moe_ops.h`
- `csrc/rocm_ops.cpp`
- `csrc/transpose_add.cu`
- `op_tests/dense_70x2.py`
- `op_tests/test_transpose_add.py`

## Provenance

- Merge commit: `1a7637a4c440`
- Captured at knowledge cutoff: 2026-05-15
- Source of truth: the upstream PR linked above. This page summarizes upstream
  metadata; consult the PR for the authoritative diff.
