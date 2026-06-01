---
id: blog-amdgpu-kernel-opt-guide
title: AMDGPU Kernel Optimization Guide (nod-ai / shark-ai)
author: Jakub Kuderski (nod-ai)
url: https://github.com/nod-ai/amd-shark-ai/blob/main/docs/amdgpu_kernel_optimization_guide.md
source_category: community-note
architectures:
- gfx942
- gfx950
tags:
- lds
- bank-conflict-avoidance
- vectorized-loads
- dpp
- permute
- wave-reduce
- occupancy-tuning
- vgpr
- agpr
- swizzle
- cdna
retrieved_at: '2026-05-30'
---

# AMDGPU Kernel Optimization Guide (nod-ai / shark-ai)

A practitioner's optimization guide for AMD GFX9 GPUs (MI300X / CDNA3 and
MI350X-MI355X / CDNA4), written by Jakub Kuderski in the context of IREE / Turbine
kernel codegen. It maps Vulkan/CUDA/AMDGPU terminology and gives concrete,
measured guidance for getting kernels to peak. The guide states it is *not* a
reference manual and is not guaranteed accurate; this page summarizes its key
points and quotes a few short figures for citation (no license is stated upstream,
so original prose is not reproduced).

## Topology, registers, memory (as reported)

- **MI300X:** 8 XCDs, 304 CUs, chiplet design; each CU has 4 SIMD16. Peak HBM3
  ≈ 5.2 TB/s. Cache: L1D 32 kB (write-through), L1I 64 kB, L2 4 MB/XCD,
  LLC/MALL 256 MB (non-coherent).
- **MI355X (CDNA4):** 8 XCDs, 2 IODs, 256 active CUs; peak HBM3E ≈ 8 TB/s; CDNA4
  adds L2 coherency optimizations.
- **Registers (MI300):** up to 104 SGPRs/workgroup, 256 VGPRs/thread, 256
  AGPRs/thread; VGPR/AGPR share a file (CDNA2+). Spilling goes **first to AGPRs**
  (`v_accvgpr_*`), **then to scratch** (`scratch_store_*`).
- **LDS:** MI300X 64 kB / 32 banks, bank index `(address / 4) % 32`. MI355X
  160 KB / 64 banks (640 × 4 B entries), `(address / 4) % 64`, 256 B/clock read.

## Cross-lane primitive latency (MI300, measured — citable)

Quantitative data this guide contributes that the wiki's `hw-cross-lane` and
`technique-wave-reduce` pages now reference:

| Primitive | Approx. cycles (incl. its `s_waitcnt`) |
|---|---|
| `ds_permute` / `ds_bpermute` | ~50 |
| `ds_swizzle` | ~50 |
| DPP | 4–12 |
| `v_permlane` (gfx950) | 4–8 |

Ordering — **speed:** `v_permlane >= DPP > ds_swizzle >= ds_permute > ds_bpermute`;
**generality** is the reverse. DPP needs no `s_waitcnt`; the LDS-based ops do.
MLIR/ROCDL ops: `rocdl.ds_bpermute`, `rocdl.ds_swizzle`, `rocdl.update.dpp` /
`amdgpu.dpp`, `rocdl.permlane*` / `amdgpu.permlane_swap`.

## Concrete recommendations (paraphrased)

1. Launch workgroups as a **multiple of the CU count** for full-GPU utilization;
   engage all **4 IODs** to approach peak bandwidth.
2. Use **workgroup size 256** (multiple) to fill all four SIMDs; 128 to save power.
3. Verify ISA dumps show **zero spilled registers**
   (`.sgpr_spill_count` / `.vgpr_spill_count`).
4. Use HIP `__launch_bounds__(MAX_THREADS_PER_BLOCK, MIN_WARPS_PER_EXECUTION_UNIT)`
   to exceed the default 128-register cap; hint the allocator with the LLVM
   `amdgpu-waves-per-eu` function attribute.
5. **Coalesce global memory:** optimal access is **16 B / 128-bit**
   (`global_load_dwordx4`); make it subgroup-contiguous so the whole subgroup
   touches **512 B at once**; up to **4 adjacent `global_load_dwordx4` form a
   clause** = one data-fabric transaction; engage all **four L1 cache sets** by
   loading 4 distinct 128 B cache lines per workgroup. Use **non-temporal**
   loads/stores for streamed, non-cached data.
6. Prefer **wide LDS instructions** (`ds_read_b128`, `ds_read2_b64`, …) and `ds_`
   over `flat_`.
7. **On MI350, prefer XOR-based swizzling over padding** to avoid LDS bank
   conflicts — the `ds_read_b128` column-wise access pattern (as in MFMA) makes
   padding hard to apply.

## Reference

- Upstream: <https://github.com/nod-ai/amd-shark-ai/blob/main/docs/amdgpu_kernel_optimization_guide.md>
- Author: Jakub Kuderski (@kuhar), nod-ai; written for IREE / Turbine kernels.
