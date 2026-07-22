---
id: ref-gfx950-validation-harness
title: ROCmKernelWiki MI355X gfx950 Validation Harness
repo: jhinpan/ROCmKernelWiki
url: https://github.com/jhinpan/ROCmKernelWiki/tree/main/validation
source_category: reference-repo
architectures:
- gfx950
- gfx942
tags:
- cdna
- async-copy
- mfma
- permute
- vgpr
- lds
retrieved_at: '2026-07-22'
languages:
- hip
- python
---

# ROCmKernelWiki MI355X gfx950 Validation Harness

The repository-local `validation/run.py` pins runtime execution to MI355X device
0 and keeps gfx942 compile-only. The retained
`validation/results/gfx950-mi355x-rocm720/` bundle records commands, compiler
acceptance/rejection, emitted ISA and HSA metadata, runtime output, source hashes,
and machine-readable verdicts.

The captured run reports 26 pass, 0 fail, and 1 source-recorded verdict under
ROCm 7.2 / clang 22. It validates device properties, the direct-to-LDS width
matrix and uniform-destination runtime behavior, permlane/f8f6f4/XF32 compiler
capabilities, and metadata extraction. It does not establish cache, partition,
LDS phase-group, numeric MXFP, or performance claims.
