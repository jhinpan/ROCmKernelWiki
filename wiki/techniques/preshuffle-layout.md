---
id: technique-preshuffle-layout
title: Pre-shuffle Weight Layout for MFMA
type: technique
architectures:
- gfx942
- gfx950
tags:
- preshuffle-layout
- swizzled-layout
- mfma
- matrix-core
- data-reuse
- lds
- fp8
- bf16
confidence: source-reported
reproducibility: snippet
hardware_features:
- mfma
- matrix-core
- lds
- ds-instructions
kernel_types:
- gemm
- fp8-gemm
- grouped-gemm
languages:
- hip
- flydsl
related:
- hw-mfma
- kernel-flydsl-preshuffle-gemm
- technique-lds-swizzling
- technique-bank-conflict-avoidance
- technique-vectorized-loads
- lang-flydsl
sources:
- ref-flydsl
- hw-mfma
- kernel-flydsl-preshuffle-gemm
- blog-matrix-cores-cdna
- ref-matrix-calculator
- doc-cdna3-isa
implemented_by:
- pr-composable_kernel-1838
- pr-composable_kernel-933
- pr-composable_kernel-2564
- pr-composable_kernel-2096
- pr-composable_kernel-1297
- pr-composable_kernel-3193
- pr-composable_kernel-2949
- pr-composable_kernel-3237
---
# Pre-shuffle Weight Layout for MFMA

## The problem: every MFMA wants a specific register layout

`v_mfma_*` instructions are executed by a whole **wavefront (64 lanes)**, and
each instruction expects the A and B tiles to be *distributed across the VGPRs of
all 64 lanes in a fixed, shape-specific pattern* (see [MFMA](../hardware/mfma.md)).
For `v_mfma_f32_16x16x16_f16`, lane `l` must supply specific `(row, k)` elements
of A and `(k, col)` elements of B — the mapping is not row-major, not
column-major, but an interleave determined by the matrix unit.

A naive GEMM loads a row-major weight tile from HBM into LDS, then each lane
reads back exactly the elements its MFMA slot needs. That read-back is a
**runtime swizzle**: a permutation that happens every K-iteration of every tile,
on every CU, for the entire life of the kernel. It costs `ds_read` traffic, often
provokes [LDS bank conflicts](../patterns/bank-conflicts.md), and burns VGPRs on
addressing math.

**Pre-shuffling** moves that permutation *off the critical path*. Weights are
static during inference, so you permute them **once** — offline or in a one-time
prologue — into the exact order the MFMA lanes consume. The hot loop then streams
contiguous bytes straight into registers (or LDS) with no index arithmetic and no
conflict-prone scatter.

## What "MFMA-friendly" actually means

For a given instruction the calculator tells you the lane/VGPR mapping. Example:

```bash
# Print exactly which (i,k) of A each lane+VGPR holds for the 16x16x16 f16 op
python3 matrix_calculator.py --architecture cdna3 \
    --instruction v_mfma_f32_16x16x16_f16 --detail-instruction
```

A pre-shuffled buffer is the original weight matrix reordered so that
**consecutive bytes in memory = consecutive elements consumed by one lane, then
the next lane**. After the reorder, the inner loop is a flat vectorized copy:

```cpp
// HIP sketch: consume a pre-shuffled B-tile with NO runtime swizzle.
// B_shuffled is laid out so lane `l` reads its 4 contiguous f16 B-elements
// for v_mfma_f32_16x16x16_f16 directly via a 64-bit load.
using f16x4 = __attribute__((__vector_size__(4 * sizeof(__fp16)))) __fp16;
using f16x4_in  = f16x4;
using float4    = __attribute__((__vector_size__(4 * sizeof(float)))) float;

__device__ void gemm_k_step(const f16x4* __restrict__ A_shuf,   // [64] lanes
                            const f16x4* __restrict__ B_shuf,   // [64] lanes
                            float4& acc)
{
    const int lane = threadIdx.x & 63;          // wave64
    // One 64-bit ds_read / global_load per lane — already in MFMA order.
    f16x4 a = A_shuf[lane];
    f16x4 b = B_shuf[lane];
    // No permute, no bank-conflict scatter: feed the matrix core directly.
    acc = __builtin_amdgcn_mfma_f32_16x16x16f16(a, b, acc, 0, 0, 0);
}
```

Contrast with the naive path, where `A_shuf[lane]`/`B_shuf[lane]` would instead be
`ds_bpermute`/strided `ds_read2` gathers reconstructing the lane order from a
row-major LDS tile on every K-step.

## Generating the shuffled layout with FlyDSL

[FlyDSL](../languages/flydsl.md) makes the layout itself a first-class object: a
`!fly.layout` is a `(Shape, Stride)` pair, and an **MFMA atom** carries the exact
lane/VGPR distribution the matrix core expects. You compose the logical tile
layout with the atom's layout to derive the offline permutation, then materialize
the weight in that order.

```python
import flyc
from fly import layout, mfma

# Logical weight tile: K-major [N, K] f16 weights.
W = layout((N, K), dtype="f16")

# The MFMA atom's B-operand layout for the 16x16x16 f16 instruction encodes
# the (lane, vgpr) -> (k, n) mapping the matrix core consumes.
atom = mfma.atom("v_mfma_f32_16x16x16_f16", operand="B")

# Compose: produce the permutation that maps logical (n,k) to MFMA-feed order.
W_shuffled = W.tile(atom.tile_shape).swizzle(atom.layout)

@flyc.jit(arch="gfx942")
def preshuffle(dst, src):
    # One-time reorder; emits contiguous stores in MFMA-consumption order.
    dst[...] = src[...].as_layout(W_shuffled)
```

The result is consumed by the [FlyDSL pre-shuffle GEMM
kernel](../kernels/flydsl-preshuffle-gemm.md), whose K-loop issues plain
`global_load`/`ds_read` in atom order and feeds the matrix core with zero runtime
swizzle. The same idea backs hipBLASLt/CK "weight-preshuffle" GEMM variants and
AITER's pre-shuffled FP8 MoE weights.

## When it pays off

Pre-shuffling is a win when **the shuffled operand is reused far more than it is
produced**:

- **Inference GEMM / GEMV** — weights are frozen. Shuffle once at model load;
  amortize across every token. Highest payoff.
- **FP8 / low-precision GEMM** — narrow types pack 4 elements into a 32-bit lane;
  the MFMA interleave for `v_mfma_f32_16x16x32_fp8_fp8` is even less
  memory-friendly, so removing the runtime swizzle saves more.
- **Grouped / MoE GEMM** — per-expert weights are static; pre-shuffle each
  expert's tile once.

It is usually **not** worth it for the *activation* operand (which is produced
fresh every layer) unless the producer can be fused to emit MFMA-order output
directly — see [kernel fusion](kernel-fusion.md).

## Costs and pitfalls

- **Storage / packaging.** The pre-shuffled weight is a different byte layout, so
  it must be produced by an offline tool or a one-time prologue and shipped with
  the model. It is **instruction-shape specific**: a buffer shuffled for
  `16x16x16` is wrong for `32x32x8`. Re-shuffle (or store per-shape) if you retune
  `matrix_instr_nonkdim`.
- **Architecture specific.** The lane mapping can differ across the MFMA shape
  set; always regenerate from the [matrix calculator](../../sources/refs/ref-matrix-calculator.md)
  or a FlyDSL atom rather than hardcoding indices. FP8 is FNUZ on gfx942 vs OCP on
  gfx950 — the *encoding* differs even when the *shuffle* matches.
- **Still want vectorization.** Pre-shuffle so each lane's elements are
  contiguous, enabling `ds_read_b64`/`ds_read_b128` and wide `global_load` — pair
  with [vectorized loads](vectorized-loads.md). A correct permutation that breaks
  128-bit alignment can erase the gain.
- **Verify, don't trust.** A wrong permutation silently computes garbage (no
  fault). Validate against a row-major reference GEMM before benchmarking.

## Relationship to runtime LDS swizzling

[LDS swizzling](lds-swizzling.md) solves the *bank-conflict* part of the same
problem but keeps the permutation **in the hot loop** (write rotated, read
rotated). Pre-shuffle is the stronger move when the operand is static: it removes
the permutation entirely instead of making it cheaper. The two compose — a
pre-shuffled tile can still be staged through LDS with a swizzle that preserves
conflict-free `ds_read` for the activation side.

## Sources

- [FlyDSL — layout DSL with MFMA atoms](https://github.com/) (`ref-flydsl`): `!fly.layout` (Shape,Stride) composition used to derive offline weight permutations.
- [MFMA — AMD Matrix Core Instructions](../hardware/mfma.md): per-wavefront register distribution that defines the target layout.
- [FlyDSL pre-shuffle GEMM kernel](../kernels/flydsl-preshuffle-gemm.md): end-to-end consumer of the shuffled layout.
- [Matrix Core Programming on CDNA3 and CDNA4](https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores-cdna/README.html): lane/VGPR operand mappings.
- [AMD Matrix Instruction Calculator](https://github.com/ROCm/amd_matrix_instruction_calculator): authoritative per-instruction layout dump.
- [CDNA3 ISA Reference Guide](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf): VOP3P-MAI MFMA encoding and operand semantics.
