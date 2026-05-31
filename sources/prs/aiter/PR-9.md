---
id: pr-aiter-9
repo: ROCm/aiter
pr: 9
title: Expose rmsnorm functions
author: ruanjm
date: '2024-12-12'
url: https://github.com/ROCm/aiter/pull/9
source_category: upstream-code
architectures:
- gfx942
tags:
- gfx942
- rmsnorm
techniques: []
hardware_features: []
kernel_types:
- rmsnorm
languages:
- hip
- cpp
- triton
captured_at: '2026-05-15'
status: merged
merge_sha: e315267736b0
inclusion_reason: keyword 'rmsnorm'
changed_paths:
- ater/__init__.py
- ater/ops/rmsnorm.py
related:
- kernel-rmsnorm
- technique-wave-reduce
- technique-kernel-fusion
---
# Expose rmsnorm functions

**Repository:** [ROCm/aiter](https://github.com/ROCm/aiter) · **PR:** [#9](https://github.com/ROCm/aiter/pull/9) · **Merged:** 2024-12-12 · **Author:** @ruanjm

**Inclusion reason:** keyword 'rmsnorm'

## Summary (from upstream PR description)

ATT

## Changed files (2 total, first 2 shown)

- `ater/__init__.py`
- `ater/ops/rmsnorm.py`

## Provenance

- Merge commit: `e315267736b0`
- Captured at knowledge cutoff: 2026-05-15
- Source of truth: the upstream PR linked above. This page summarizes upstream
  metadata; consult the PR for the authoritative diff.
