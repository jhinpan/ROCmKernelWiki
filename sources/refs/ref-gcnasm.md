---
id: ref-gcnasm
title: gcnasm — GCN Assembly & HIP Programming Examples
repo: carlushuang/gcnasm
url: https://github.com/carlushuang/gcnasm
author: carlushuang
source_category: reference-repo
architectures:
- gfx942
- gfx950
tags:
- gcn-asm
- mfma
- bandwidth-bench
- dpp
- async-copy
languages:
- gcn-asm
- hip
retrieved_at: '2026-05-15'
---

# gcnasm — GCN Assembly & HIP Programming Examples

A collection of AMD GPU programming examples (CDNA/RDNA, primarily
gfx942/MI300) covering hand-written GCN assembly kernels, HIP device code, and
PyTorch/Triton extensions. Standout examples: `bandwidth_memread` (float4
non-temporal persistent bandwidth microbench, ~4.56 TB/s on MI308X);
`vector_add_asm` (persistent kernel, `buffer_load_dword ... offen lds` async load
to LDS, double LDS buffering, OOB-based control flow, `vmcnt(3)` pipelining);
`matrix_core` / `matrix_core_gfx950` (MFMA demos); `hgemm` (128×128 MFMA asm);
`wave_reduce_dpp`, `ds_permute`, `transpose-lds`. An excellent low-level
MFMA/assembly reference.

## Reference

- Upstream: <https://github.com/carlushuang/gcnasm>
