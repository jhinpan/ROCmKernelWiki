---
id: pr-Tensile-9
repo: ROCm/Tensile
pr: 9
title: Tensor Contractions in OpenCL
author: guacamoleo
date: '2016-04-22'
url: https://github.com/ROCm/Tensile/pull/9
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
- gcn-asm
- python
captured_at: '2026-05-15'
status: merged
merge_sha: 869ca1210cf0
inclusion_reason: kernel path 'CobaltGen/CobaltGenBackend.py'; keyword 'gemm'
changed_paths:
- .gitignore
- CMakeLists.txt
- CobaltBenchmark/CMakeLists.txt
- CobaltBenchmark/CobaltBenchmark.cpp
- CobaltBenchmark/CobaltBenchmark.h
- CobaltGen/CobaltGenBackend.py
- CobaltGen/CobaltGenBenchmark.py
- CobaltGen/FileReader.py
- CobaltGen/FileWriter.py
- CobaltGen/KernelWriter.py
- CobaltGen/README.md
- CobaltGen/SolutionCandidateGenerator.py
- CobaltGen/SolutionSelectionWriter.py
- CobaltGen/SolutionSet.py
- CobaltGen/SolutionWriter.py
- CobaltGen/Structs.py
- CobaltLib/CMakeLists.txt
- CobaltLib/FindOpenCL.cmake
- CobaltLib/README.md
- CobaltLib/include/Cobalt.h
---

# Tensor Contractions in OpenCL

**Repository:** [ROCm/Tensile](https://github.com/ROCm/Tensile) · **PR:** [#9](https://github.com/ROCm/Tensile/pull/9) · **Merged:** 2016-04-22 · **Author:** @guacamoleo

**Inclusion reason:** kernel path 'CobaltGen/CobaltGenBackend.py'; keyword 'gemm'

## Summary (from upstream PR description)

this commit provides implementation for tensor contractions in OpenCL.

## Changed files (45 total, first 20 shown)

- `.gitignore`
- `CMakeLists.txt`
- `CobaltBenchmark/CMakeLists.txt`
- `CobaltBenchmark/CobaltBenchmark.cpp`
- `CobaltBenchmark/CobaltBenchmark.h`
- `CobaltGen/CobaltGenBackend.py`
- `CobaltGen/CobaltGenBenchmark.py`
- `CobaltGen/FileReader.py`
- `CobaltGen/FileWriter.py`
- `CobaltGen/KernelWriter.py`
- `CobaltGen/README.md`
- `CobaltGen/SolutionCandidateGenerator.py`
- `CobaltGen/SolutionSelectionWriter.py`
- `CobaltGen/SolutionSet.py`
- `CobaltGen/SolutionWriter.py`
- `CobaltGen/Structs.py`
- `CobaltLib/CMakeLists.txt`
- `CobaltLib/FindOpenCL.cmake`
- `CobaltLib/README.md`
- `CobaltLib/include/Cobalt.h`

## Provenance

- Merge commit: `869ca1210cf0`
- Captured at knowledge cutoff: 2026-05-15
- Source of truth: the upstream PR linked above. This page summarizes upstream
  metadata; consult the PR for the authoritative diff.
