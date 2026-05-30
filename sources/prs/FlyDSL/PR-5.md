---
id: pr-FlyDSL-5
repo: ROCm/FlyDSL
pr: 5
title: add build_llvm script and add softmax/rmsnorm/layernorm ops test
author: XingerZhu
date: '2025-12-02'
url: https://github.com/ROCm/FlyDSL/pull/5
source_category: upstream-code
architectures:
- gfx942
tags:
- gfx942
- layernorm
- rmsnorm
- softmax
techniques: []
hardware_features: []
kernel_types:
- rmsnorm
- layernorm
- softmax
languages:
- flydsl
- mlir
- python
captured_at: '2026-05-15'
status: merged
merge_sha: 615f208aff33
inclusion_reason: keyword 'rmsnorm'
changed_paths:
- README.md
- build.sh
- build_llvm.sh
- tests/python/gpu/test_layernorm.py
- tests/python/gpu/test_rmsnorm.py
- tests/python/gpu/test_softmax.py
---

# add build_llvm script and add softmax/rmsnorm/layernorm ops test

**Repository:** [ROCm/FlyDSL](https://github.com/ROCm/FlyDSL) · **PR:** [#5](https://github.com/ROCm/FlyDSL/pull/5) · **Merged:** 2025-12-02 · **Author:** @XingerZhu

**Inclusion reason:** keyword 'rmsnorm'

## Summary (from upstream PR description)

Motivation

Technical Details

Test Plan

Test Result

Submission Checklist

 Look over the contributing guidelines at https://github.com/ROCm/ROCm/blob/develop/CONTRIBUTING.md#pull-requests.

## Changed files (6 total, first 6 shown)

- `README.md`
- `build.sh`
- `build_llvm.sh`
- `tests/python/gpu/test_layernorm.py`
- `tests/python/gpu/test_rmsnorm.py`
- `tests/python/gpu/test_softmax.py`

## Provenance

- Merge commit: `615f208aff33`
- Captured at knowledge cutoff: 2026-05-15
- Source of truth: the upstream PR linked above. This page summarizes upstream
  metadata; consult the PR for the authoritative diff.
