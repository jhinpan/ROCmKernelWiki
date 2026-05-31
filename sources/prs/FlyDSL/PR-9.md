---
id: pr-FlyDSL-9
repo: ROCm/FlyDSL
pr: 9
title: perftest support functions
author: zhiding512
date: '2025-12-04'
url: https://github.com/ROCm/FlyDSL/pull/9
source_category: upstream-code
architectures:
- gfx942
tags:
- gfx942
- quantization
techniques:
- fine-grained-quantization
hardware_features: []
kernel_types:
- quantization
languages:
- flydsl
- mlir
- python
captured_at: '2026-05-15'
status: merged
merge_sha: a9b2440e74e9
inclusion_reason: keyword 'quant'
changed_paths:
- tests/benchmark/per_token_quant_benchmark.py
- tests/utils.py
facet_source: inferred
related:
- technique-fine-grained-quantization
---
# perftest support functions

**Repository:** [ROCm/FlyDSL](https://github.com/ROCm/FlyDSL) · **PR:** [#9](https://github.com/ROCm/FlyDSL/pull/9) · **Merged:** 2025-12-04 · **Author:** @zhiding512

**Inclusion reason:** keyword 'quant'

## Summary (from upstream PR description)

_No PR description provided upstream._

## Changed files (2 total, first 2 shown)

- `tests/benchmark/per_token_quant_benchmark.py`
- `tests/utils.py`

## Provenance

- Merge commit: `a9b2440e74e9`
- Captured at knowledge cutoff: 2026-05-15
- Source of truth: the upstream PR linked above. This page summarizes upstream
  metadata; consult the PR for the authoritative diff.
