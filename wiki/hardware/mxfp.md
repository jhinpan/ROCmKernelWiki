---
id: hw-mxfp
title: MXFP — Block-Scaled FP8/FP6/FP4 Microscaling (CDNA4 gfx950)
type: hardware
version_sensitive:
- vs-fp8-ocp-gfx950
- vs-tf32-dropped-gfx950
architectures:
- gfx950
tags:
- mxfp
- block-scale
- fp8
- fp6
- fp4
- mfma
- matrix-core
confidence: verified
evidence_basis:
- source_id: doc-cdna4-isa
  evidence_type: official-doc
- source_id: doc-cdna4-whitepaper
  evidence_type: official-doc
- source_id: blog-matrix-cores-cdna
  evidence_type: upstream-code
related:
- hw-mfma
- migration-gfx942-to-gfx950
- kernel-fp8-gemm
- technique-fine-grained-quantization
- lang-rocwmma
sources:
- doc-cdna4-isa
- doc-cdna4-whitepaper
- blog-matrix-cores-cdna
- blog-fp8-gemm-cdna4
- ref-matrix-calculator
aliases:
- MXFP
- microscaling
- block-scaled FP8
- MXFP4
- MXFP6
- f8f6f4
- E8M0
implemented_by:
- pr-composable_kernel-3603
- pr-composable_kernel-2297
- pr-composable_kernel-3601
- pr-composable_kernel-2152
- pr-composable_kernel-2000
- pr-vllm-42952
- pr-composable_kernel-2665
- pr-FlyDSL-191
---
# MXFP — Block-Scaled FP8/FP6/FP4 Microscaling (CDNA4 gfx950)

## Overview

**MXFP** ("MX" = microscaling) is the Open Compute Project (OCP) family of
narrow floating-point formats in which a small **block** of low-precision
elements shares a single power-of-two **scale**. CDNA4 (gfx950, MI350X/MI355X)
is the first AMD Instinct architecture to implement these formats natively in
the matrix core, with dedicated conversion ops and block-scaled
[MFMA](mfma.md) instructions. The combination is what lets MI355X advertise
**MXFP6/MXFP4 at 10 PFLOPS dense** — double the OCP-FP8 rate.

A microscaled tensor is two arrays:

- **Elements** — `K` narrow values (FP8, FP6, or FP4) per block.
- **Scale** — one **E8M0** value per block that all `K` elements are multiplied
  by during the dot product. The OCP MX spec fixes the block size at **32
  elements** along the reduction (K) dimension.

This decouples *dynamic range* (carried by the per-block exponent scale) from
*precision* (carried by the narrow mantissa), which is what makes 4- and 6-bit
inference numerically viable.

## The element formats (CBSZ/BLGP codes)

CDNA4 unifies FP8/FP6/FP4 behind a single MMA opcode, `f8f6f4`. The legacy
`CBSZ` (matrix-A) and `BLGP` (matrix-B) modifier fields are **repurposed as
per-matrix format selectors**, and A and B may use different formats:

| Code | Format | Bits | Exp/Mant | Notes |
|---|---|---|---|---|
| `000` | E4M3 | 8 | 4 exp / 3 mant | OCP FP8 |
| `001` | E5M2 | 8 | 5 exp / 2 mant | OCP BF8 |
| `010` | E2M3 | 6 | 2 exp / 3 mant | FP6 |
| `011` | E3M2 | 6 | 3 exp / 2 mant | BF6 |
| `100` | E2M1 | 4 | 2 exp / 1 mant | FP4 |

> **OCP, not FNUZ.** CDNA4 FP8 is the **OCP** encoding (with Inf/NaN), *not* the
> **FNUZ** encoding used by CDNA3 (gfx942). Weights quantized for one are not
> bit-compatible with the other — see
> [gfx942 → gfx950 migration](../migration/gfx942-to-gfx950.md).

## The E8M0 block scale

The shared scale is **E8M0**: 8 bits, *all exponent*, no sign and no mantissa,
with **bias 127** — i.e. it encodes a pure power of two `2^(s-127)`. The all-ones
encoding (`0xFF`) is reserved as NaN. Because the scale is a power of two,
applying it is an exponent add, not a multiply, which is why the scaled MFMA
costs the same as the unscaled one.

For a single block the effective dot-product contribution is:

```
acc += scaleA * scaleB * Σ_{k=0..31} A_k * B_k
     = 2^(sA-127) * 2^(sB-127) * Σ A_k * B_k
```

## Block-scaled MFMA instructions

CDNA4 adds an unscaled unified path and a microscaled path:

```ptx
; Unified FP8/FP6/FP4 dense MMA, FP32 accumulate, NO scale
v_mfma_f32_16x16x128_f8f6f4   a[...], b[...], c[...]
v_mfma_f32_32x32x64_f8f6f4    a[...], b[...], c[...]

; Microscaled (MX) variants: take per-block E8M0 scale operands
v_mfma_scale_f32_16x16x128_f8f6f4  a[...], b[...], c[...], scaleA, scaleB
v_mfma_scale_f32_32x32x64_f8f6f4   a[...], b[...], c[...], scaleA, scaleB
```

Note the K dimensions: **128** for the 16×16 shape and **64** for the 32×32
shape. With the OCP block size of 32, that is 4 and 2 scale blocks along K per
instruction, respectively. Scaling is gated by `ABID[0]`: with `ABID[0]=1` the
provided E8M0 scales are applied; with `ABID[0]=0` all scales are forced to
`1.0` (so the same opcode degenerates to the unscaled behaviour).

The LLVM/Clang intrinsic for the 16×16 scaled form is:

```cpp
// gfx950 only. acc, A, B are wavefront-distributed register tiles.
// opselA/opselB pick the byte lane of the packed E8M0 scale; the two trailing
// immediates carry the A-format and B-format selectors (see table above).
using f32x4  = __attribute__((__vector_size__(16))) float;
using i32x8  = __attribute__((__vector_size__(32))) int;  // packed narrow A/B

__device__ f32x4 mx_mma_16x16x128(i32x8 a, i32x8 b, f32x4 acc,
                                  int scaleA /*E8M0*/, int scaleB /*E8M0*/)
{
    // llvm.amdgcn.mfma.scale.f32.16x16x128.f8f6f4
    return __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
        a, b, acc,
        /*cbsz=A-fmt*/ 0, /*blgp=B-fmt*/ 0,    // 0 = E4M3 for both
        /*opselA*/ 0, scaleA,
        /*opselB*/ 0, scaleB);
}
```

In practice you should let [Composable Kernel](../languages/composable-kernel.md),
hipBLASLt, or the [Triton AMD backend](../languages/triton-amd.md) emit these —
the packed register layout for FP6 (6 bits straddling dword boundaries) is
intricate. Use the [Matrix Instruction Calculator](../../sources/refs/ref-matrix-calculator.md)
to dump the exact element→register mapping:

```bash
python3 matrix_calculator.py --architecture cdna4 \
    --instruction v_mfma_scale_f32_16x16x128_f8f6f4 --detail-instruction
```

## Conversion / packing ops

Producing MX tensors needs format conversion with the E8M0 scale folded in.
CDNA4 adds scale-aware pack converters, including stochastic-rounding (`_sr_`)
variants for quantization-aware training:

```ptx
v_cvt_scalef32_pk_fp4_f32      ; pack FP32 -> FP4 with an FP32 scale
v_cvt_scalef32_pk32_fp6_f32    ; pack 32 FP32 lanes -> FP6 block
v_cvt_scalef32_sr_pk_fp4_f32   ; same, stochastic rounding
```

A typical quantization step computes `amax` over the 32-element block, derives
the E8M0 exponent `s = ceil(log2(amax / fmax_format))`, then converts each
element scaled by `2^-(s-127)`. Because the scale is an exponent, this is
exact in the high bits and avoids a divide.

## Throughput

From the [CDNA4 whitepaper](../../sources/docs/doc-cdna4-whitepaper.md):

| Format | MI355X dense | Relative to OCP-FP8 |
|---|---|---|
| OCP-FP8 (E4M3/E5M2) | 5.0 PFLOPS | 1× |
| MXFP6 / BF6 | 10 PFLOPS | 2× |
| MXFP4 | 10 PFLOPS | 2× |
| FP16/BF16 | 2.5 PFLOPS | 0.5× |

The 2× step from FP8 to FP6/FP4 comes directly from the wider-K opcodes
(`16x16x128` vs the FP8 `16x16x32`): more reduction work per issued instruction
at the same issue rate. Note the silicon trade — to make room for the MX path,
CDNA4 **dropped the native TF32/XF32 matrix path** (now emulated via BF16) and
**halved per-CU FP64 matrix** throughput vs CDNA3.

## Practical notes

- **Block alignment.** Keep the K tiling a multiple of 32 so each MX block maps
  cleanly to one E8M0 scale; misalignment forces scalar fix-up of partial
  blocks.
- **Scale layout.** Scales are a separate, much smaller tensor (1 byte per 32
  elements ≈ 3% overhead for FP8, more impactful for FP4). Stage them so the MMA
  can read `scaleA`/`scaleB` without extra global traffic.
- **Accuracy.** FP4 (E2M1) has a 1-bit mantissa; per-block E8M0 scaling is what
  keeps it usable. For accuracy-sensitive layers prefer MXFP6, or mix formats
  (e.g. FP6 activations × FP4 weights) using the per-matrix CBSZ/BLGP selectors.

## See also

- [MFMA — AMD Matrix Core Instructions](mfma.md)
- [FP8 block-scaled GEMM kernel](../kernels/fp8-gemm.md)
- [Fine-grained quantization technique](../techniques/fine-grained-quantization.md)
- [gfx942 → gfx950 migration](../migration/gfx942-to-gfx950.md)

## Sources

- [AMD CDNA4 ISA Reference Guide](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-cdna4-instruction-set-architecture.pdf)
- [AMD CDNA4 Architecture Whitepaper](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-4-architecture-whitepaper.pdf)
- [Matrix Core Programming on CDNA3 and CDNA4](https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores-cdna/README.html)
- [OCP Microscaling Formats (MX) Specification v1.0](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)
