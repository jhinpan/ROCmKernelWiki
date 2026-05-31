---
id: pr-FlyDSL-7
repo: ROCm/FlyDSL
pr: 7
title: use rocir in softmax/layernorm/rmsnorm ops
author: XingerZhu
date: '2025-12-03'
url: https://github.com/ROCm/FlyDSL/pull/7
source_category: upstream-code
architectures:
- gfx942
tags:
- gfx942
- layernorm
- lds
- mfma
- rmsnorm
- softmax
techniques:
- mfma-pipelining
hardware_features:
- lds
- mfma
kernel_types:
- layernorm
- rmsnorm
- softmax
languages:
- flydsl
- mlir
- python
captured_at: '2026-05-15'
status: merged
merge_sha: 76f4d290121d
inclusion_reason: kernel path 'tests/python/gpu/test_layernorm.py'; keyword 'mfma'
changed_paths:
- tests/python/gpu/test_layernorm.py
- tests/python/gpu/test_rmsnorm.py
- tests/python/gpu/test_softmax.py
facet_source: inferred
related:
- technique-wave-reduce
- kernel-flash-attention-ck
- kernel-rmsnorm
---
# use rocir in softmax/layernorm/rmsnorm ops

**Repository:** [ROCm/FlyDSL](https://github.com/ROCm/FlyDSL) · **PR:** [#7](https://github.com/ROCm/FlyDSL/pull/7) · **Merged:** 2025-12-03 · **Author:** @XingerZhu

**Inclusion reason:** kernel path 'tests/python/gpu/test_layernorm.py'; keyword 'mfma'

## Summary (from upstream PR description)

Motivation

Technical Details

Test Plan

Test Result
========================================================================
Test Summary
MLIR IR Tests (Lowering):        22/22 passed
Python IR Tests (Generation):    15/15 passed
Example Tests (ROCDL):           0/0 passed
GPU Execution Tests:             13/13 passed
Benchmark Tests:                 2/2 passed
Verified Capabilities:
✓ Rocir IR generation and lowering
✓ Coordinate operations (crd2idx, layouts)
✓ ROCDL dialect operations (381 ops exposed)
✓ GPU kernel compilation (MLIR → HSACO)
✓ GPU kernel execution (HIP runtime)
✓ Shared memory optimizations (LDS)
✓ MFMA operations (Pure Python API)
✓ Performance benchmarking (bandwidth tests)
Submission Checklist

 Look over the contributing guidelines at https://github.com/ROCm/ROCm/blob/develop/CONTRIBUTING.md#pull-requests.

## Changed files (3 total, first 3 shown)

- `tests/python/gpu/test_layernorm.py`
- `tests/python/gpu/test_rmsnorm.py`
- `tests/python/gpu/test_softmax.py`

## Provenance

- Merge commit: `76f4d290121d`
- Captured at knowledge cutoff: 2026-05-15
- Source of truth: the upstream PR linked above. This page summarizes upstream
  metadata; consult the PR for the authoritative diff.
