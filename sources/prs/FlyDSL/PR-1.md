---
id: pr-FlyDSL-1
repo: ROCm/FlyDSL
pr: 1
title: Add pass to python
author: coderfeli
date: '2025-11-25'
url: https://github.com/ROCm/FlyDSL/pull/1
source_category: upstream-code
architectures:
- gfx942
tags:
- gemm
- gfx942
techniques: []
hardware_features: []
kernel_types:
- gemm
languages:
- flydsl
- mlir
- python
captured_at: '2026-05-15'
status: merged
merge_sha: 270108c44bd5
inclusion_reason: kernel path 'tests/python/examples/test_gpu_gemm.py'; keyword 'gemm'
changed_paths:
- build.sh
- python/rocdsl/__init__.py
- python/rocdsl/cute_opt_runner.py
- python/rocdsl/passes.py
- python_bindings/CMakeLists.txt
- python_bindings/CuteDialectModule.cpp
- python_bindings/CutePassesModule.cpp
- python_bindings/cute.py
- run_tests.sh
- tests/README.md
- tests/python/conftest.py
- tests/python/examples/test_gpu_gemm.py
- tests/python/test_arith_operators.py
- tests/python/test_basic_ops.py
- tests/python/test_cute_basic.py
- tests/python/test_cute_divide.py
- tests/python/test_cute_local.py
- tests/python/test_cute_product.py
- tests/python/test_local_ops.py
- tests/python/test_product_divide.py
---

# Add pass to python

**Repository:** [ROCm/FlyDSL](https://github.com/ROCm/FlyDSL) · **PR:** [#1](https://github.com/ROCm/FlyDSL/pull/1) · **Merged:** 2025-11-25 · **Author:** @coderfeli

**Inclusion reason:** kernel path 'tests/python/examples/test_gpu_gemm.py'; keyword 'gemm'

## Summary (from upstream PR description)

Motivation

Technical Details

Test Plan

Test Result

Submission Checklist

 Look over the contributing guidelines at https://github.com/ROCm/ROCm/blob/develop/CONTRIBUTING.md#pull-requests.

## Changed files (20 total, first 20 shown)

- `build.sh`
- `python/rocdsl/__init__.py`
- `python/rocdsl/cute_opt_runner.py`
- `python/rocdsl/passes.py`
- `python_bindings/CMakeLists.txt`
- `python_bindings/CuteDialectModule.cpp`
- `python_bindings/CutePassesModule.cpp`
- `python_bindings/cute.py`
- `run_tests.sh`
- `tests/README.md`
- `tests/python/conftest.py`
- `tests/python/examples/test_gpu_gemm.py`
- `tests/python/test_arith_operators.py`
- `tests/python/test_basic_ops.py`
- `tests/python/test_cute_basic.py`
- `tests/python/test_cute_divide.py`
- `tests/python/test_cute_local.py`
- `tests/python/test_cute_product.py`
- `tests/python/test_local_ops.py`
- `tests/python/test_product_divide.py`

## Provenance

- Merge commit: `270108c44bd5`
- Captured at knowledge cutoff: 2026-05-15
- Source of truth: the upstream PR linked above. This page summarizes upstream
  metadata; consult the PR for the authoritative diff.
