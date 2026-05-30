---
id: ref-matrix-calculator
title: AMD Matrix Instruction Calculator
repo: ROCm/amd_matrix_instruction_calculator
url: https://github.com/ROCm/amd_matrix_instruction_calculator
author: ROCm
source_category: reference-repo
architectures:
- gfx942
- gfx950
tags:
- mfma
- matrix-core
- agpr
languages:
- python
retrieved_at: '2026-05-15'
---

# AMD Matrix Instruction Calculator

An official tool that, given an MFMA/WMMA instruction, reports its
shape, supported dtypes, register usage (Arch vs Acc VGPRs), operand
element→register mapping, FLOP counts, and per-CU throughput. Indispensable for
deriving exact MFMA operand layouts instead of hand-computing them. Supports
`--list-instructions`, `--detail-instruction`, `--get-register`, `--matrix-entry`.

## Reference

- Upstream: <https://github.com/ROCm/amd_matrix_instruction_calculator>
