---
id: migration-gfx942-to-gfx950
title: Migrating Kernels CDNA3 (gfx942) → CDNA4 (gfx950)
type: migration
version_sensitive:
- vs-fp8-fnuz-gfx942
- vs-fp8-ocp-gfx950
- vs-lds-size-gfx950
- vs-permlane16-gfx950
- vs-tf32-dropped-gfx950
architectures:
- gfx942
- gfx950
tags:
- cdna
- fp8
- mxfp
- fp6
- fp4
- block-scale
- lds
- mfma
- permute
- bf16
- tf32
confidence: source-reported
cross_vendor_note: 'This is a cross-*generation* port within AMD CDNA, not a cross-vendor
  one, but the failure modes rhyme with a vendor port: a binary that ran correctly
  on gfx942 can compile and launch on gfx950 yet produce wrong numbers. The sharp
  edge is FP8 — CDNA3 uses AMD''s OCP-incompatible FNUZ (E4M3/E5M2) encoding while
  CDNA4 uses standard OCP FP8 — so FP8 weights/activations and their scales are NOT
  bit-portable across the two. Native TF32/XF32 matrix support is also gone on
  gfx950—the gfx942 intrinsic does not lower, so software must explicitly select
  a supported datatype/path—and FP64 matrix throughput per CU is halved. Treat
  every affected numeric path as requiring re-validation, not just recompilation.

  '
related:
- hw-mfma
- hw-mxfp
- hw-lds
- hw-cross-lane
- hw-async-copy-lds
- migration-cuda-to-hip
sources:
- doc-cdna4-isa
- doc-cdna4-whitepaper
- hw-mxfp
- hw-mfma
- hw-lds
reproducibility: snippet
languages:
- hip
- gcn-asm
aliases:
- MI300 to MI355X
- CDNA3 to CDNA4
- FP8 portability
- FNUZ to OCP
implemented_by:
- pr-FlyDSL-191
- pr-aiter-2136
- pr-aiter-2491
- pr-composable_kernel-3603
- pr-composable_kernel-2152
- pr-FlyDSL-554
- pr-FlyDSL-278
- pr-FlyDSL-153
---
# Migrating Kernels CDNA3 (gfx942) → CDNA4 (gfx950)

## Overview

CDNA4 (gfx950, MI350X/MI355X) is source-compatible with CDNA3 (gfx942,
MI300A/X/325X) for the vast majority of HIP code: rebuild with
`--offload-arch=gfx950` and most kernels run. The danger is the minority that
**silently** change behaviour. This page is a checklist of what actually differs
at the ISA/hardware level and how to adapt a kernel that was tuned for gfx942.

The big-ticket changes:

| Area | gfx942 (CDNA3) | gfx950 (CDNA4) | Action on port |
|---|---|---|---|
| FP8 encoding | **FNUZ** (E4M3/E5M2, OCP-incompatible) | **OCP** FP8 (E4M3/E5M2) | Re-quantize / re-validate numerics |
| MX block formats | none | FP6/FP4 + E8M0 block scales (`f8f6f4`) | New opt path; opt-in |
| LDS size | 64 KB/CU | 160 KB/CU | Can grow tiles / occupancy |
| LDS banks | 32 banks × 512 dwords | 64 banks × 640 dwords | Re-check swizzle/padding |
| Direct-to-LDS width | ≤4 B (one dword) | ≤16 B (`global_load_lds_dwordx4`; also `dwordx3`) | Wider async copies |
| `v_permlane16/32_swap` | absent | present | Faster cross-lane reductions |
| TF32/XF32 matrix | native MFMA path | **dropped**; gfx942 intrinsic does not lower | Explicitly select BF16 or another supported path |
| FP64 matrix | full rate | **halved per CU** | Re-budget FP64-heavy kernels |

## 1. FP8 numerics — the one that bites silently

This is the single most important item. CDNA3 FP8 is **FNUZ** ("finite, NaN
unsigned zero"): no infinities, a single NaN bit pattern, and a different
exponent bias than the OCP spec. CDNA4 implements **OCP FP8** (E4M3 / E5M2 per
the Open Compute microscaling spec). The two are *not* bit-compatible — the same
8-bit pattern decodes to a different real value.

Consequences for a port:

- **Stored FP8 weights/KV-cache are not portable.** A checkpoint quantized to
  FP8 on MI300X must be re-quantized (or dequantized→requantized) for MI355X.
- **Scale factors change.** Because the bias and dynamic range differ, the
  per-tensor/per-block scale that kept values in-range on FNUZ is wrong for OCP.
- **Library types differ.** hipBLASLt exposes distinct FP8 datatypes for the two
  encodings; select the gfx950 OCP type, do not reuse the FNUZ enum.

In HIP, gate the type by arch rather than hardcoding:

```cpp
#include <hip/hip_fp8.h>

// CDNA3 (gfx942): FNUZ FP8.   CDNA4 (gfx950): OCP FP8.
#if defined(__gfx942__)
  using fp8_e4m3 = __hip_fp8_e4m3_fnuz;   // FNUZ (CDNA3 device pass)
#else
  using fp8_e4m3 = __hip_fp8_e4m3;        // OCP (gfx950 device pass + the host pass)
#endif

__device__ float decode(fp8_e4m3 x) { return float(x); }  // bias handled by type
```

> Recompiling does **not** fix stored data. The compiler picks the right encode/
> decode for the target, but bytes already on disk were produced under the other
> encoding. Re-quantize at the data layer.

## 2. New MX low-precision path (opportunity, not a port hazard)

CDNA4 adds the unified `f8f6f4` MFMA family and microscaled (MX) variants —
covered in depth on the [MXFP page](../hardware/mxfp.md) and the
[MFMA page](../hardware/mfma.md). These are *new capability*: existing FP16/BF16/
FP8 kernels keep working unchanged. To exploit them you opt in to the new
instructions:

```ptx
; gfx950-only: unified FP8/FP6/FP4 dense MMA, FP32 accumulate
v_mfma_f32_16x16x128_f8f6f4   a[...], b[...], c[...]
; microscaled: one E8M0 (8-bit exp, bias 127) scale per MX block
v_mfma_scale_f32_16x16x128_f8f6f4 ...
```

The `CBSZ`/`BLGP` modifier fields are repurposed as per-matrix format selectors
(`000`=E4M3, `001`=E5M2, `010`=E2M3/FP6, `011`=E3M2/BF6, `100`=E2M1/FP4); mixed
A/B formats are legal. MXFP6/MXFP4 roughly double the matrix throughput of FP8 on
MI355X (10 PF vs 5 PF dense, per the [CDNA4 whitepaper](../../sources/docs/doc-cdna4-whitepaper.md)).

## 3. LDS: bigger, more banks

CDNA4 grows LDS from **64 KB/CU (32 banks)** to **160 KB/CU (64 banks)** — see
the [LDS page](../hardware/lds.md). Two practical effects:

- **More room.** Tile sizes / multi-buffering depths that were LDS-capped on
  gfx942 can grow, or you can raise occupancy. If you computed LDS-per-workgroup
  to hit a target waves/CU, recompute it for the new budget.
- **Bank count doubled (32 → 64).** Any hand-tuned padding or swizzle that
  assumed a 32-bank conflict pattern must be re-derived. A stride that was
  conflict-free on 32 banks may conflict on 64, and vice-versa. Re-run the
  conflict analysis or your padding constant against the new geometry.

```cpp
// Don't bake in 64KB. Query the limit and size LDS from device props.
hipDeviceProp_t p; hipGetDeviceProperties(&p, dev);
size_t lds_bytes = p.sharedMemPerBlock;     // 64KB-class on gfx942, larger on gfx950
// ... choose TILE_K / buffering depth so per-block LDS <= lds_bytes
```

## 4. Direct-to-LDS widens to 16 bytes

The direct-to-LDS async copy (HBM→LDS bypassing VGPRs — AMD's analog of
NVIDIA `cp.async`; see [async copy](../hardware/async-copy-lds.md)) gains
12-/16-byte transfers on gfx950 (`global_load_lds_dwordx3/x4`,
`llvm.amdgcn.load.to.lds`). A gfx942 pipeline limited to **single-dword (4 B)**
direct-to-LDS copies can move 4× as much per instruction by switching to
`dwordx4` on gfx950, improving the streaming/compute overlap in GEMM and
attention prologues. (Verified on MI350X, ROCm 7.2: the legal
`__builtin_amdgcn_load_to_lds` byte sizes are `{1, 2, 4}` on gfx942 vs
`{1, 2, 4, 12, 16}` on gfx950 — there is **no** 8 B / `dwordx2` form on either
generation.)

## 5. Cross-lane: `v_permlane16_swap` now available

The lane-swap op `v_permlane16_swap_b32` (and `v_permlane32_swap_b32`) is
**gfx950-only** (absent on gfx942) — see
[cross-lane ops](../hardware/cross-lane.md). On gfx942 a full wave reduction
typically does DPP within 16-lane rows, then `ds_swizzle`/`ds_bpermute` for the
cross-row step. On gfx950 you can replace compatible LDS-crossbar hops with the
`_swap` builtins. Each returns the swapped versions of its two inputs, so when
both inputs are the same value the partner is in element 1 for a lane in the
lower half and element 0 for a lane in the upper half. Keep the gfx942 path
under an arch guard:

```cpp
__device__ float add_partner_row16(float v, int lane) {
#if defined(__gfx950__)
    // One 16-lane-half butterfly, with no LDS-unit traffic.
    auto sw = __builtin_amdgcn_permlane16_swap(
        __float_as_int(v), __float_as_int(v), false, false);
    int partner = (lane & 16) ? sw[0] : sw[1];
    return v + __int_as_float(partner);
#else
    // gfx942: pull the corresponding lane from the other 16-lane half.
    int partner = __builtin_amdgcn_ds_bpermute((lane ^ 16) << 2,
                                                __float_as_int(v));
    return v + __int_as_float(partner);
#endif
}
```

This helper is one butterfly step and assumes every lane already holds a valid
row partial. A complete wave64 reduction also needs the analogous 32-lane-half
step (`permlane32_swap` on gfx950, or `ds_bpermute` with `lane ^ 32` on gfx942).

> **Do not use** `__builtin_amdgcn_permlanex16(v, v, sel0, sel1, …)` here: that
> SGPR-selector form is an RDNA (gfx10+) instruction and fails to compile for
> gfx950 with `needs target feature gfx10-insts` (verified on MI350X, ROCm 7.2).
> CDNA4 exposes only the `_swap` form.

## 6. TF32/XF32 dropped; FP64 matrix halved

- **TF32 (XF32) has no native matrix path on CDNA4.** gfx942's
  `v_mfma_f32_16x16x8_xf32` / `32x32x4` are gone. With ROCm 7.2 / clang 22 the
  gfx942 intrinsic fails during gfx950 instruction selection (`Cannot select`);
  there is no automatic BF16 fallback guarantee. Port the kernel to an explicit
  BF16 or other supported path so the precision/throughput tradeoff is visible
  and intentional.
- **FP64 matrix throughput per CU is halved** on gfx950 (silicon reallocated to
  MX formats). MI355X still lists 78.6 TF FP64 vs the relatively higher
  per-CU rate on MI300X — FP64-matrix-bound kernels (mostly HPC, not ML) need a
  fresh roofline before assuming they scale with the new part.

## Migration checklist

1. Rebuild with `--offload-arch=gfx950`; resolve any `__gfx942__`-guarded code.
2. **Re-quantize all FP8 data** and recompute scales for OCP encoding; pick the
   OCP hipBLASLt FP8 type. Validate numerics, don't just check it runs.
3. Recompute per-block LDS budget against 160 KB/64 banks; re-derive swizzle/
   padding for 64 banks.
4. (Optional) Widen direct-to-LDS copies to `dwordx4`; adopt `permlane16` for
   reductions; adopt `f8f6f4`/MX MFMA for new low-precision throughput.
5. Audit TF32 and FP64-matrix paths: switch TF32 → explicit BF16; re-roofline
   FP64 kernels for the halved matrix rate.

## Sources

- [AMD CDNA4 ISA Reference Guide](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-cdna4-instruction-set-architecture.pdf)
- [AMD CDNA4 Architecture Whitepaper](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-4-architecture-whitepaper.pdf)
- [MXFP / block-scaled formats (this wiki)](../hardware/mxfp.md)
- [MFMA matrix instructions (this wiki)](../hardware/mfma.md)
- [LDS sizes and banks (this wiki)](../hardware/lds.md)
