---
id: hw-mfma
title: MFMA — AMD Matrix Core Instructions (CDNA)
type: hardware
version_sensitive:
- vs-fp8-fnuz-gfx942
- vs-fp8-ocp-gfx950
- vs-cdna-unified-vgpr-agpr-allocation
- vs-tf32-dropped-gfx950
architectures:
- gfx942
- gfx950
tags:
- mfma
- matrix-core
- agpr
- fp8
- bf16
confidence: verified
evidence_basis:
- source_id: doc-cdna3-isa
  evidence_type: official-doc
- source_id: doc-cdna4-isa
  evidence_type: official-doc
- source_id: ref-matrix-calculator
  evidence_type: upstream-code
related:
- hw-mxfp
- hw-wavefront
- technique-mfma-pipelining
sources:
- doc-cdna3-isa
- doc-cdna4-isa
- blog-amd-matrix-cores
- blog-matrix-cores-cdna
- ref-matrix-calculator
aliases:
- MFMA
- matrix cores
- matrix fused multiply add
- xdlops
implemented_by:
- pr-composable_kernel-2110
- pr-triton-368
- pr-triton-358
- pr-composable_kernel-2202
- pr-composable_kernel-1902
- pr-Tensile-827
- pr-composable_kernel-2199
---
# MFMA — AMD Matrix Core Instructions (CDNA)

## Overview

`v_mfma_*` (Matrix Fused Multiply-Add) is the family of AMD CDNA tensor-core
instructions that compute `D = A · B + C` for small matrix tiles. Unlike a
normal VALU op, an MFMA is issued and executed cooperatively by an **entire
wavefront (64 lanes)** — the elements of A, B, C and D are distributed across
the VGPRs of all 64 lanes. MFMA is the AMD analog of NVIDIA's `wgmma`/`tcgen05`
tensor-core path, and historically was nicknamed **XDLOP**.

The instruction is encoded as VOP3P-MAI. The HIP/Clang builtin form is:

```cpp
// d = a * b + c, computed across the whole wavefront
// CDfmt = accumulator/output format, ABfmt = input format
// cbsz/abid/blgp are broadcast/block-select modifiers (legacy multi-block ops)
using float4 = __attribute__((__vector_size__(4 * sizeof(float)))) float;

__device__ float4 mfma_16x16x16_f16(
    const __attribute__((__vector_size__(4 * sizeof(__fp16)))) __fp16 a,
    const __attribute__((__vector_size__(4 * sizeof(__fp16)))) __fp16 b,
    float4 c)
{
    // 16x16x16 FP16 -> FP32: each lane holds 4 A-halves, 4 B-halves, 4 C-floats
    return __builtin_amdgcn_mfma_f32_16x16x16f16(a, b, c, 0, 0, 0);
}
```

Most kernels do **not** call the builtin directly. Prefer
[`rocWMMA`](../languages/rocwmma.md) (a portable C++ fragment API) or let
[Composable Kernel](../languages/composable-kernel.md) / hipBLASLt / Triton emit
MFMA for you. Use the builtin (or hand assembly) only when you need full control
of the register layout.

## Accumulation register names (AGPRs)

CDNA exposes two architectural vector-register views: **ArchVGPRs** and
**AccVGPRs (AGPRs)**. On CDNA3 and CDNA4, MFMA A, B, C, and D operands may each
use either view, but the accumulator (C/D) is conventionally kept in AGPR names
to free ArchVGPR names for addressing and to allow the matrix unit to co-issue
with the VALU. These are not independent capacity banks: up to 256 names exist
in each view, while both draw from the same combined 512-entry-per-lane physical
allocation and occupancy budget. Heavy MFMA tiling is therefore still
register-bound, so accumulator-tile size directly trades against occupancy —
see [register pressure](../patterns/vgpr-pressure.md).

## Shapes and dtypes — CDNA3 (gfx942)

MFMA shapes are written `M x N x K`. The primary dense shapes on gfx942:

| Input dtype | Acc/out | Primary shapes (M×N×K) | Example mnemonic |
|---|---|---|---|
| FP16 | FP32 | 16×16×16, 32×32×8 | `v_mfma_f32_16x16x16_f16` |
| BF16 | FP32 | 16×16×16, 32×32×8 | `v_mfma_f32_32x32x8_bf16` |
| FP8/BF8 (any mix) | FP32 | 16×16×32, 32×32×16 | `v_mfma_f32_16x16x32_fp8_fp8` |
| INT8 | INT32 | 16×16×32, 32×32×16 | `v_mfma_i32_16x16x32_i8` |
| XF32 (TF32) | FP32 | 16×16×8, 32×32×4 | `v_mfma_f32_16x16x8_xf32` |
| FP32 | FP32 | 16×16×4, 32×32×2 | `v_mfma_f32_16x16x4_f32` |
| FP64 | FP64 | 16×16×4, 4×4×4 | `v_mfma_f64_16x16x4_f64` |

Throughput grows as K widens for narrower types: `v_mfma_f32_16x16x16_f16`
delivers 8192 FLOPs in 16 cycles (2048 FLOPs/CU/cycle), while
`v_mfma_f32_16x16x32_fp8_fp8` delivers 16384 FLOPs in 16 cycles
(4096 FLOPs/CU/cycle) — exactly the 2× FP8-over-FP16 ratio reflected in the
[MI300X peak figures](../../sources/docs/doc-mi300x-datasheet.md).

> **FP8 is FNUZ on gfx942.** CDNA3's FP8 (`fp8`=E4M3, `bf8`=E5M2) uses AMD's
> *FNUZ* (OCP-incompatible) encoding — no infinities, single NaN, different
> exponent bias. It is **not** bit-compatible with the OCP FP8 introduced on
> CDNA4. This matters for any cross-architecture weight reuse — see
> [gfx942 → gfx950 migration](../migration/gfx942-to-gfx950.md).

Structured 4:2 sparsity is available through the `v_smfmac_*` family.

## Shapes and dtypes — CDNA4 (gfx950)

CDNA4 keeps most of the CDNA3 set, drops the native TF32/XF32 matrix path, and
adds a **unified low-precision** path plus microscaling. The headline new
instructions:

```ptx
; Unified FP8/FP6/FP4 dense MMA, FP32 accumulate (no scale)
v_mfma_f32_16x16x128_f8f6f4  a[...], b[...], c[...]
v_mfma_f32_32x32x64_f8f6f4   a[...], b[...], c[...]

; Microscaling (MX) variants: per-block E8M0 scale operands
v_mfma_scale_f32_16x16x128_f8f6f4 ...
v_mfma_scale_f32_32x32x64_f8f6f4  ...
```

For the `f8f6f4` ops the legacy `CBSZ`/`BLGP` modifier fields are **repurposed
as per-matrix element-format selectors** (matrix A = CBSZ, matrix B = BLGP), and
mixed A/B formats are allowed:

| Code | Format |
|---|---|
| `000` | E4M3 (FP8) |
| `001` | E5M2 (BF8) |
| `010` | E2M3 (FP6) |
| `011` | E3M2 (BF6) |
| `100` | E2M1 (FP4) |

The scaled variants take an **E8M0** (8-bit exponent, bias 127) block scale per
MX block; `ABID[0]=1` enables scaling (with `ABID[0]=0` all scales are forced to
1.0). The full block-scaling story is on the [MXFP page](mxfp.md).

Also new on gfx950: wider-K halves `v_mfma_f32_16x16x32_f16/bf16`,
`v_mfma_f32_32x32x16_f16/bf16`, `v_mfma_i32_16x16x64_i8`. Note that the native
**TF32/XF32 matrix path was dropped** on CDNA4: the gfx942 XF32 intrinsic does
not lower for gfx950, so software must explicitly choose BF16 or another
supported path. FP64 matrix throughput per CU was halved — the silicon was
reallocated to the MX formats.

## Deriving the exact register layout

Operand element→register mappings are fiddly and shape-specific. Rather than
hand-deriving them, use the official
[AMD Matrix Instruction Calculator](../../sources/refs/ref-matrix-calculator.md):

```bash
# What registers hold A[i][k] for a given instruction?
python3 matrix_calculator.py --architecture cdna3 \
    --instruction v_mfma_f32_16x16x16_f16 --detail-instruction
```

## Performance claims

See [MI300X datasheet](../../sources/docs/doc-mi300x-datasheet.md) and the
[CDNA4 whitepaper](../../sources/docs/doc-cdna4-whitepaper.md):

- MI300X: FP16/BF16 1307 TFLOPS, FP8 2615 TFLOPS, INT8 2615 TOPS (dense).
- MI355X: FP16/BF16 2.5 PF, OCP-FP8 5.0 PF, MXFP6/MXFP4 10 PF (dense).

## See also

- [MXFP / block-scaled FP8-FP6-FP4](mxfp.md)
- [MFMA pipelining technique](../techniques/mfma-pipelining.md)
- [rocWMMA language guide](../languages/rocwmma.md)
- [WMMA vs MFMA migration](../migration/wmma-vs-mfma.md)

## Sources

- [CDNA3 ISA Reference Guide](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf)
- [CDNA4 ISA Reference Guide](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-cdna4-instruction-set-architecture.pdf)
- [Matrix Core Programming on CDNA3 and CDNA4](https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores-cdna/README.html)
