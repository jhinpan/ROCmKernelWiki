---
id: pr-triton-1
repo: ROCm/triton
pr: 1
title: Enable rocm
author: micmelesse
date: '2021-08-23'
url: https://github.com/ROCm/triton/pull/1
source_category: upstream-code
architectures:
- gfx942
tags:
- gfx942
techniques: []
hardware_features: []
kernel_types: []
languages:
- triton
- mlir
captured_at: '2026-05-15'
status: merged
merge_sha: 42585e7f4806
inclusion_reason: kernel path 'include/triton/driver/kernel.h'; keyword 'kernel'
changed_paths:
- .gitmodules
- CMakeLists.txt
- include/triton/driver/backend.h
- include/triton/driver/buffer.h
- include/triton/driver/context.h
- include/triton/driver/device.h
- include/triton/driver/dispatch.h
- include/triton/driver/error.h
- include/triton/driver/handle.h
- include/triton/driver/kernel.h
- include/triton/driver/module.h
- include/triton/driver/platform.h
- include/triton/driver/stream.h
- include/triton/external/CUDA/cuda.h
- include/triton/tools/rocm_helper.h
- lib/codegen/analysis/layout.cc
- lib/codegen/analysis/swizzle.cc
- lib/codegen/pass.cc
- lib/codegen/selection/generator.cc
- lib/codegen/transform/peephole.cc
---

# Enable rocm

**Repository:** [ROCm/triton](https://github.com/ROCm/triton) · **PR:** [#1](https://github.com/ROCm/triton/pull/1) · **Merged:** 2021-08-23 · **Author:** @micmelesse

**Inclusion reason:** kernel path 'include/triton/driver/kernel.h'; keyword 'kernel'

## Summary (from upstream PR description)

This PR enables trition on rocm. There is still more work to do but we are able to compile on rocm and run an empty kernel. This is the first of many prs.
This pr uses a lot of ifdefs to work around some of the limitations of the current version of hipify torch. We can remove them as hipify torch is updated.
That being said, in order for this PR to work, this PR in hipify torch must be merged ROCm/hipify_torch#7. It basically removes bad unicode characters from source code
The major changes are in the following files

include/triton/driver/dispatch.h
lib/driver/dispatch.cc
lib/driver/module.cc
CMakeLists.txt

## Changed files (36 total, first 20 shown)

- `.gitmodules`
- `CMakeLists.txt`
- `include/triton/driver/backend.h`
- `include/triton/driver/buffer.h`
- `include/triton/driver/context.h`
- `include/triton/driver/device.h`
- `include/triton/driver/dispatch.h`
- `include/triton/driver/error.h`
- `include/triton/driver/handle.h`
- `include/triton/driver/kernel.h`
- `include/triton/driver/module.h`
- `include/triton/driver/platform.h`
- `include/triton/driver/stream.h`
- `include/triton/external/CUDA/cuda.h`
- `include/triton/tools/rocm_helper.h`
- `lib/codegen/analysis/layout.cc`
- `lib/codegen/analysis/swizzle.cc`
- `lib/codegen/pass.cc`
- `lib/codegen/selection/generator.cc`
- `lib/codegen/transform/peephole.cc`

## Provenance

- Merge commit: `42585e7f4806`
- Captured at knowledge cutoff: 2026-05-15
- Source of truth: the upstream PR linked above. This page summarizes upstream
  metadata; consult the PR for the authoritative diff.
