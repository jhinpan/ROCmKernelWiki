---
id: blog-amdgpu-kernel-opt-guide
title: AMDGPU Kernel Optimization Guide (nod-ai / shark-ai)
author: Jakub Kuderski (nod-ai)
url: https://github.com/nod-ai/amd-shark-ai/blob/efa471aeef66a260c85983cc41e833bfa769dade/docs/amdgpu_kernel_optimization_guide.md
source_category: community-note
architectures:
- gfx942
- gfx950
tags:
- lds
- bank-conflict-avoidance
- vectorized-loads
- buffer-instructions
- global-instructions
- dpp
- permute
- wave-reduce
- occupancy-tuning
- vgpr
- agpr
- swizzle
- l2-cache
- xcd
- hbm3
- mlir
- profiling
- cdna
retrieved_at: '2026-07-20'
---

# AMDGPU Kernel Optimization Guide (nod-ai / shark-ai)

This page tracks the complete set of topics in Jakub Kuderski's practitioner
guide for MI300X/CDNA3 and MI350X/MI355X/CDNA4 kernel code generation. The
source explicitly describes itself as a useful approximation rather than an
official reference. Its prose and images are not reproduced here because the
upstream document does not state a documentation license; the facts and advice
below are paraphrased and routed to canonical wiki pages.

The captured snapshot is commit
[`efa471ae`](https://github.com/nod-ai/amd-shark-ai/commit/efa471aeef66a260c85983cc41e833bfa769dade)
(retrieved 2026-07-20). The repository changed this document on 2026-07-09 to
link the empirical LDS results, although the heading inside the guide still
says `Last Update: 2025-08-14`. Use the
[`main` version](https://github.com/nod-ai/amd-shark-ai/blob/main/docs/amdgpu_kernel_optimization_guide.md)
to check for later edits.

## Coverage map

| Guide material | Canonical wiki page(s) |
|---|---|
| Vulkan/CUDA/AMDGPU terms; workgroup/CU/SIMD placement | `hw-wavefront` |
| MI300X and MI355X topology, HBM bandwidth, cache hierarchy | `hw-chiplet-xcd` |
| SGPR/VGPR/AGPR limits, spills, launch bounds, occupancy | `hw-wavefront`, `technique-occupancy-tuning`, `technique-vgpr-budgeting` |
| LDS geometry, instruction-specific phase groups, conflicts, waterfall arbitration, swizzling | `hw-lds` |
| 128-bit global accesses, clauses, L1-set coverage, non-temporal hints | `technique-vectorized-loads` |
| Buffer descriptors, 32-bit addressable window, predicated OOB behavior | `hw-memory-instructions`, `technique-buffer-oob-guard` |
| DPP, `ds_swizzle`, `ds_permute`/`ds_bpermute`, `v_permlane`, MLIR lowering | `hw-cross-lane`, `technique-wave-reduce` |

## Architecture facts reported by the guide

- **MI300X:** 8 XCDs on four IODs, 38 active of 40 physical CUs per XCD,
  304 active CUs total, and four SIMD16 units per CU. The guide calls the base
  dies “AID/IOD pairs”; the canonical page follows AMD's four-IOD wording.
  Eight HBM3 stacks expose an 8192-bit aggregate interface. AMD's direct
  calculation is `8192 bits × 5.2 Gb/s per pin ÷ 8 = 5.3248 TB/s` (normally
  quoted as 5.3 TB/s), rather than relying on the guide's “1300 MHz QDR” model.
- **MI355X:** 8 XCDs on two IODs, four Shader Engine arrays and 36 physical / 32
  active CUs per XCD, for 256 active CUs. Eight 1024-bit HBM3E interfaces at
  8 Gb/s give an 8 TB/s peak.
- In the normal non-`tgsplit` execution mode, a workgroup remains on one CU for
  its lifetime. LLVM AMDGPU also documents a gfx942/gfx950 target-specific `tgsplit`
  mode in which waves of one workgroup may be distributed across CUs, so the
  guide's absolute wording is not universal. That mode does not allocate local
  memory and its shaders cannot use the local address space (LDS); it is not a
  placement alternative for ordinary shared-memory workgroups. Each wave is
  assigned to one SIMD. A wave64 VALU instruction on SIMD16 normally issues
  over four 16-lane quarters. Up to 16 waves may belong to one workgroup. A
  256-thread workgroup contains four wave64 waves and is a useful starting
  point; the 128-thread suggestion is a power heuristic, not a universal
  performance rule.

The detailed, architecture-scoped cache geometry and coherence qualifications
are maintained in [`hw-chiplet-xcd`](../../wiki/hardware/chiplet-xcd.md). AMD's
whitepapers support L2 miss coalescing before traffic enters the data fabric.
The guide's separate claim that L2 is flushed between kernel launches is
incorrect: AMD's gfx942 memory model describes dispatch-boundary invalidation of
volatile vector/scalar **L1** lines and explicit L2 maintenance operations, not
a blanket launch-boundary L2 flush.

## Registers and occupancy reported by the guide

- Registers are DWORDs. The guide's 104-SGPR per-workgroup statement is wrong:
  the general scalar namespace is `s0`–`s101` (102 names). The raw per-wave
  `.sgpr_count` also includes enabled target-special pairs such as VCC,
  FLAT_SCRATCH, and XNACK and is not rounded. GFX9 encodes allocation in
  16-register blocks, so it can allocate 112. AMDHSA **User SGPR** is a separate
  ABI term for at most 16 dispatch-initialized registers, not a synonym for the
  general SGPR count.
- The guide reports regular indices `v0`–`v255` and accumulator indices
  `a0`–`a255`. These are per-lane architectural register names used by a wave,
  not “per-thread versus per-wave” allocation domains. On CDNA2+, their per-wave
  allocation shares one 512-entry-per-lane SIMD capacity. HSA metadata
  `.vgpr_count` is already the combined total on gfx942 and gfx950 and is
  rounded to 8; `.agpr_count` is a subset. Separate lower-level gfx942 compiler
  remarks use `round_up(round_up(NumVgprs, 4) + NumAgprs, 8)`. Occupancy is
  `floor(512 / vector_alloc)`, capped at eight waves/SIMD. Exactly 256 combined
  entries therefore allow two waves, while a legal total above 256 can run as
  one wave without necessarily spilling.
- Its metadata example uses `.sgpr_spill_count`, `.vgpr_spill_count`,
  `.agpr_count`, `.vgpr_count`, and `.private_segment_fixed_size`. Do not round
  separate low-level gfx942 VGPR and AGPR components independently to eight,
  and never add `.agpr_count` to metadata `.vgpr_count`. Remapping a regular
  live range into an unused AGPR
  index is register allocation inside the unified vector file, not a spill.
  Real spill evidence is scratch/private-segment allocation plus emitted
  scratch traffic; always inspect the final metadata and ISA.
- `__launch_bounds__(MAX_THREADS_PER_BLOCK, MIN_WARPS_PER_EXECUTION_UNIT)` and
  the LLVM/Clang `amdgpu-waves-per-eu` attribute trade register budget against
  target occupancy. The guide spells the latter once as `amgpu-waves-per-eu`;
  that is a typo. Its separate statement that 128 registers is the default cap
  is compiler/snapshot-specific, not a CDNA3/CDNA4 architectural limit.

The guide's **10 waves/SIMD (40 waves/CU)** claim is not valid for CDNA3/CDNA4.
Real gfx950 devices report 2048 work-items/CU: `2048 / wave64 = 32 waves/CU =
4 × 8 waves/SIMD`. The wiki uses the measured 8/32 ceilings throughout and
retains the guide value only as a documented correction.

## LDS material reported by the guide

- gfx942: 64 KiB/CU, 32 four-byte banks, bank index
  `(byte_address / 4) % 32`.
- gfx950: 160 KiB/CU, 64 four-byte banks, bank index
  `(byte_address / 4) % 64`, with a reported 256 B/clock read datapath.
- A same-address read is a broadcast. A bank conflict requires different
  addresses mapping to the same bank **within the same instruction-specific
  phase group**. For gfx942 the guide separately reports an address-send limit
  of 16 addresses/SIMD/cycle and up to 32/CU/cycle; that unverified front-end
  figure is not a universal phase-group rule. The guide describes conflict
  arbitration as choosing the lowest-thread-id non-conflicting subset and then
  replaying leftovers, producing a one-lane-at-a-time waterfall in the worst
  case.
- Prefer aligned wide `ds_read_b128`, `ds_read2_b64`, and `ds_write_b128`
  operations and explicit `ds_*` addressing over generic `flat_*` when the
  address space is known. On gfx950 MFMA-style column reads, XOR swizzling is
  generally easier to tune than padding.

The canonical page also corrects a detail not stated accurately by generic
summaries: LDS allocation units are 512 bytes on gfx942 and 1280 bytes on
gfx950, so occupancy math must round a workgroup's request for its target.

The exact gfx942/gfx950 phase groups, including the non-contiguous b128 lane
sets, live in [`hw-lds`](../../wiki/hardware/lds.md) and are linked to the
upstream
[`empirical-lds` results](https://github.com/nod-ai/amd-shark-ai/blob/efa471aeef66a260c85983cc41e833bfa769dade/docs/empirical-lds/lds_summary.md).
The guide's ranges ending at `T64` are endpoint typos; a wave64's final lane is
`T63`.

## Global and buffer-memory recommendations

- Prefer aligned 16-byte `global_load/store_dwordx4` operations. One full
  wave64 payload is **64 × 16 B = 1 KiB**. The guide's nearby 512-byte figure
  instead describes wave64×8 B or wave32×16 B and must not be attached to a
  wave64 dwordx4 operation.
- Keep lane addresses contiguous, issue several operations before waiting, and
  spread work over every active CU/XCD/IOD. The guide reports that up to four
  adjacent dwordx4 loads can form a clause and reduce data-fabric transactions;
  treat that as target/compiler-sensitive until confirmed from final ISA and
  counters.
- The four 128-byte L1D lines selected by a workgroup should cover the four L1D
  sets. Non-temporal is a target-specific cache-policy hint for one-use data; it
  does not disable memory coherency or promise that every cache level is
  bypassed.
- Buffer addressing provides a 32-bit offset/extent window of approximately
  **4 GiB per descriptor**. This is not a device-allocation size limit: rebase
  or chunk a larger allocation into descriptor windows, or use 64-bit global
  addresses with software predication.

## Cross-lane operations and MLIR lowering

The guide labels its latency table as a Fused Softmax measurement on MI300
(including any required wait), but it also inserts a CDNA4-only `v_permlane`
row while leaving that instruction's own section as a TODO. Preserve the
evidence boundary explicitly:

| Primitive | Approximate cycles | Wait | Reach | Evidence scope |
|---|---:|---|---|---|
| `ds_permute` / `ds_bpermute` | ~50 | `s_waitcnt` | arbitrary wave64 | guide-reported MI300 measurement |
| `ds_swizzle` | ~50 | `s_waitcnt` | fixed 32-lane pattern | guide-reported MI300 measurement |
| DPP | 4–12 | none | fixed row/broadcast pattern | guide-reported MI300 measurement |
| `v_permlane` family | 4–8 | none | architecture-specific pattern | guide estimate of unclear provenance; not an MI300 instruction-set measurement |

`ds_permute`/`ds_bpermute` and `ds_swizzle` use the LDS crossbar without
allocating LDS storage. DPP is a VALU source modifier and does not consume the
LDS unit. The guide's speed ordering is `v_permlane >= DPP > ds_swizzle >=
ds_permute > ds_bpermute`; flexibility is roughly the reverse. It calls the
CDNA modifier **DPP**, while RDNA manuals use **DPP8/DPP16** names and different
target-specific encodings.

At the captured LLVM/MLIR level, `amdgpu.dpp` is an enum-friendly wrapper around
`rocdl.update.dpp`. Lowering may first produce `v_mov_b32_dpp`; LLVM's
`GCNDPPCombine` can fuse it into a following compatible VALU instruction (for
example `v_add_f32_dpp`). Non-default `row_mask` or `bank_mask` values can block
that combine, so final ISA inspection is mandatory. The guide maps
`ds_bpermute` and `ds_swizzle` to ROCDL operations but notes that a dedicated
`rocdl.ds_permute` operation was not available in its snapshot.

## Resource links carried by the guide

Official material:

- [MI300/CDNA3 ISA](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf)
- [MI350/CDNA4 ISA](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-cdna4-instruction-set-architecture.pdf)
- [CDNA3 whitepaper](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-3-white-paper.pdf)
- [CDNA4 whitepaper](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-4-architecture-whitepaper.pdf)
- [ROCm 6.1.1 LLM fine-tuning and inference optimization (historical page linked by the guide)](https://rocm.docs.amd.com/en/docs-6.1.1/how-to/llm-fine-tuning-optimization/index.html)
- [ROCm 7.2.4 inference-optimization successor](https://rocm.docs.amd.com/en/docs-7.2.4/how-to/rocm-for-ai/inference-optimization/index.html)
- [MI300 compute and memory partitioning modes](https://rocm.blogs.amd.com/software-tools-optimization/compute-memory-modes/README.html)

Additional background linked upstream:

- [ORNL Introduction to AMDGPU](https://www.olcf.ornl.gov/wp-content/uploads/2019/10/ORNL_Application_Readiness_Workshop-AMD_GPU_Basics.pdf)
- [Chips and Cheese CDNA3 architecture overview](https://chipsandcheese.com/2023/12/17/amds-cdna-3-compute-architecture/)
- [Chips and Cheese MI300X measurements](https://chipsandcheese.com/2024/06/25/testing-amds-giant-mi300x/)

Technical links embedded elsewhere in the captured guide:

- [IREE](https://iree.dev) and [IREE Turbine](https://github.com/iree-org/iree-turbine)
- [ROCm GPU architecture specifications](https://rocm.docs.amd.com/en/latest/reference/gpu-arch-specs.html)
- [LLVM AMDGPU IR attributes (`amdgpu-waves-per-eu`)](https://llvm.org/docs/AMDGPUUsage.html#llvm-ir-attributes)
- [ROCm Blogs — avoiding LDS bank conflicts](https://rocm.blogs.amd.com/software-tools-optimization/lds-bank-conflict/README.html)
- [Pinned empirical LDS measurements](https://github.com/nod-ai/amd-shark-ai/blob/efa471aeef66a260c85983cc41e833bfa769dade/docs/empirical-lds/lds_summary.md)
- [GPUOpen — GCN assembly cross-lane operations](https://gpuopen.com/learn/amd-gcn-assembly-cross-lane-operations/)
- [RDNA3 ISA reference for DPP8/DPP16](https://www.amd.com/content/dam/amd/en/documents/radeon-tech-docs/instruction-set-architectures/rdna3-shader-instruction-set-architecture-feb-2023_0.pdf)
- [`GCNDPPCombine::combineDPPMov` at the guide's pinned LLVM revision](https://github.com/llvm/llvm-project/blob/ab51eccf88f5321e7c60591c5546b254b6afab99/llvm/lib/Target/AMDGPU/GCNDPPCombine.cpp#L522)
- [`rocprofv2`/rocprofiler link used for the guide's latency table](https://github.com/ROCm/rocprofiler?tab=readme-ov-file#plugin-support)

## Reference

- Captured upstream snapshot: <https://github.com/nod-ai/amd-shark-ai/blob/efa471aeef66a260c85983cc41e833bfa769dade/docs/amdgpu_kernel_optimization_guide.md>
- Live upstream document: <https://github.com/nod-ai/amd-shark-ai/blob/main/docs/amdgpu_kernel_optimization_guide.md>
- Author: Jakub Kuderski (@kuhar), nod-ai; written from IREE/Turbine kernel work.
