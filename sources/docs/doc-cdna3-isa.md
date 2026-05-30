---
id: doc-cdna3-isa
title: AMD Instinct MI300 (CDNA3) Instruction Set Architecture Reference Guide
url: https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf
source_category: official-doc
architectures:
- gfx942
tags:
- mfma
- lds
- buffer-instructions
- s-waitcnt
- wave64
- cdna
retrieved_at: '2026-05-15'
---

# AMD Instinct MI300 (CDNA3) Instruction Set Architecture Reference Guide

The authoritative ISA reference for CDNA3 (gfx942 / MI300 family), revision
5-August-2025. Documents the VOP3P-MAI MFMA encoding, the LDS (64 kB/CU, 32
banks of 512 Dwords, 32-bit wide), MUBUF/MTBUF buffer instructions with their
128-bit resource descriptor (V#) and out-of-bounds semantics, FLAT/GLOBAL/SCRATCH
addressing, the `s_waitcnt` counters (VMCNT 6-bit, LGKMCNT 4-bit, EXPCNT 3-bit),
the direct-to-LDS load path, and cross-lane primitives (`ds_swizzle`,
`ds_permute`/`ds_bpermute`, DPP modifiers).

Key facts used across this wiki:

- LDS: *"64 kB memory per compute unit, segmented into 32 banks of 512 Dwords,
  each bank being 32 bits wide."* Bank conflicts serialize; an access can take
  2–64 cycles depending on conflicts.
- Buffer OOB: *"Reads that go out-of-range return zero ... Writes that are
  out-of-range do not write anything."* — the basis for branchless boundary
  guards on AMD.
- FP8 on CDNA3 is the **FNUZ** (OCP-incompatible) encoding — numerically
  distinct from the OCP FP8 introduced on CDNA4.

## Reference

- Upstream: <https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf>
