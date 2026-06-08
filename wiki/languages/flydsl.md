---
id: lang-flydsl
title: FlyDSL — Python + MLIR Layout DSL for AMD Kernels
type: language
architectures:
- gfx942
- gfx950
- gfx1201
- gfx1250
tags:
- flydsl
- mlir
- rocdl
- mfma
- matrix-core
- preshuffle-layout
- swizzled-layout
- python
confidence: experimental
reproducibility: snippet
languages:
- flydsl
- mlir
- python
kernel_types:
- gemm
- fp8-gemm
- fused-moe
- flash-attention
- attention
related:
- hw-mfma
- technique-preshuffle-layout
- kernel-flydsl-preshuffle-gemm
- kernel-flydsl-flash-attention
- lang-composable-kernel
- lang-triton-amd
sources:
- ref-flydsl
- ref-flydsl-kernel-profiling
- ref-aiter
- doc-llvm-amdgpu
- blog-amd-matrix-cores
- doc-cdna3-isa
implemented_by:
- pr-aiter-2726
- pr-aiter-2581
- pr-aiter-2497
- pr-aiter-2390
- pr-aiter-2113
- pr-aiter-1561
- pr-composable_kernel-2320
- pr-composable_kernel-2040
---
# FlyDSL — Python + MLIR Layout DSL for AMD Kernels

## Overview

**FlyDSL** is an experimental, Apache-2.0 Python + MLIR domain-specific language
for writing AMD GPU kernels in terms of **algebraic tensor layouts** rather than
raw pointer arithmetic. It centers on a custom MLIR dialect, **`fly`**, whose
core type is a layout `!fly.layout` describing a `(Shape, Stride)` pair — the same
hierarchical layout algebra popularized by CuTe/CUTLASS, retargeted to the AMD
toolchain. A kernel is authored in Python, lowered through the `fly` dialect to
**ROCDL** (the AMDGPU LLVM IR dialect), then to LLVM IR, and finally compiled to
a HIP **fatbin** that you launch like any other device binary.

FlyDSL targets the matrix cores directly through **MFMA atoms** (and WMMA on
RDNA4), so it is a natural fit for GEMM-shaped work. It supports
**gfx942 (CDNA3)**, **gfx950 (CDNA4)**, **gfx1250 (CDNA-next)**, and
**gfx1201 (RDNA4)**. In practice it has shown up as an optional backend inside
[AITER](../../sources/refs/ref-aiter.md) for fused-MoE expert GEMMs, where
[pre-shuffled weight layouts](../techniques/preshuffle-layout.md) matter.

> **Status: experimental.** APIs, dialect ops, and pass names change frequently.
> Treat the snippets below as illustrative of the programming model, not as a
> pinned, version-stable interface. Prefer [Composable Kernel](composable-kernel.md)
> or the [Triton AMD backend](triton-amd.md) for production kernels today.

## The layout algebra: `Shape`, `Stride`, and `crd2idx`

A layout maps a logical coordinate to a linear memory offset. As in CuTe, a
FlyDSL layout is a (possibly nested) `Shape` paired with a matching `Stride`; the
map from a coordinate to an index is the classic `crd2idx`:

```text
idx = crd2idx(coord, shape, stride) = Σ_i  coord_i * stride_i      (row-major: stride = suffix-product of shape)
```

A column-major 128×64 tile and its layout, expressed in the Python frontend:

```python
import fly
from fly import Shape, Stride, Layout

# A 128 x 64 tile, column-major (leading dim = 128)
A = Layout(Shape(128, 64), Stride(1, 128))

assert A.crd2idx((0, 0)) == 0
assert A.crd2idx((1, 0)) == 1       # step down a column
assert A.crd2idx((0, 1)) == 128     # step across a row

# Hierarchical (tiled) layout: split M=128 into (8 lanes-of-16) x (16)
At = Layout(Shape((16, 8), 64), Stride((1, 16), 128))
print(At.crd2idx(((3, 2), 5)))      # nested coord -> flat offset
```

Because the layout is data, not control flow, FlyDSL can *compose* and *invert*
layouts to derive register-to-LDS and LDS-to-global maps mechanically — exactly
the bookkeeping that is error-prone when hand-written for MFMA register
distributions. This is what makes [pre-shuffle](../techniques/preshuffle-layout.md)
weight packing expressible as a layout transform instead of an index hack.

## Authoring a kernel: `@flyc.kernel` and `@flyc.jit`

Two decorators drive code generation:

- **`@flyc.kernel`** marks a Python function as a device kernel (the `fly`-dialect
  entry point). Its body is traced into MLIR rather than executed eagerly.
- **`@flyc.jit`** JIT-compiles a host launcher: it specializes the kernel for the
  current target arch, runs the lowering pipeline, caches the fatbin, and returns
  a callable you invoke with a launch grid.

```python
import fly
from fly import flyc, Shape, Stride, Layout
from fly.atom import MfmaAtom  # 16x16x16 f16->f32 matrix-core atom

# A block-level GEMM tile: C[BM,BN] += A[BM,BK] * B[BK,BN]
@flyc.kernel
def gemm_tile(a_ptr, b_ptr, c_ptr, BM: int, BN: int, BK: int):
    # MFMA atom selects the matrix-core instruction + its register layout
    mma = MfmaAtom("v_mfma_f32_16x16x16_f16")

    # Layouts describe how the global tiles map to memory
    a = Layout(Shape(BM, BK), Stride(1, BM))   # col-major A
    b = Layout(Shape(BK, BN), Stride(BN, 1))   # row-major B

    acc = fly.zeros(mma.c_layout)              # accumulator in AGPRs
    for k in fly.range(0, BK, mma.K):
        ra = fly.load(a_ptr, a.slice(k_axis=k))   # -> VGPRs
        rb = fly.load(b_ptr, b.slice(k_axis=k))
        acc = mma(ra, rb, acc)                     # D = A*B + C across the wave
    fly.store(c_ptr, acc, Layout(Shape(BM, BN), Stride(1, BM)))


@flyc.jit(arch="gfx942")
def gemm(a, b, c, M, N, K):
    grid = (M // 128, N // 128)
    gemm_tile[grid](a, b, c, BM=128, BN=128, BK=64)
```

`MfmaAtom` is the key abstraction: it bundles an MFMA instruction (see
[the MFMA page](../hardware/mfma.md)) with the fragment layouts for A, B and the
accumulator, so `acc` is automatically allocated in **AGPRs** and `fly.load`
produces the correct per-lane register distribution. On gfx950 you would select a
wider-K or `f8f6f4` atom; on gfx1201 the atom maps to a `v_wmma_*` instruction.

## The lowering pipeline: Fly → ROCDL → LLVM → fatbin

`@flyc.jit` runs a fixed MLIR pipeline. Conceptually:

```text
  Python trace
      │  (build fly-dialect module)
      ▼
  fly        : !fly.layout ops, fly.load/store/mma, crd2idx folding, tiling
      │  --fly-lower-to-rocdl  (atoms → amdgcn.mfma intrinsics; layouts → addressing)
      ▼
  rocdl/llvm : llvm.amdgcn.mfma.*, llvm.amdgcn.ds.* , buffer/global addressing
      │  llc -mcpu=gfx942  (AMDGPU backend)
      ▼
  amdgcn obj → clang-offload-bundler → fatbin  (loaded via HIP module API)
```

You can dump the IR at each stage, which is the main debugging workflow:

```bash
# Emit the fly-dialect module, then the ROCDL-lowered form
flyc gemm.py --emit=fly    -o gemm.fly.mlir
flyc gemm.py --emit=rocdl  -o gemm.rocdl.mlir   --arch gfx942

# Confirm the matrix-core intrinsic survived lowering
grep -n "amdgcn.mfma" gemm.rocdl.mlir
#   %d = llvm.call @llvm.amdgcn.mfma.f32.16x16x16f16(%a, %b, %c, 0, 0, 0)
```

The ROCDL stage is where correctness is easiest to verify: the emitted
`llvm.amdgcn.mfma.*` / `llvm.amdgcn.ds.*` intrinsics are documented in the
[LLVM AMDGPU backend](../../sources/docs/doc-llvm-amdgpu.md), so a quick `grep`
tells you the intended matrix instruction and addressing mode were selected
before you ever launch.

## Why a layout IR on AMD

MFMA register distributions are notoriously fiddly — each shape scatters A/B/C
elements across the 64 lanes of a wavefront differently (see
[MFMA](../hardware/mfma.md)). FlyDSL's value proposition is that the *same*
layout algebra describes:

1. the global→LDS staging copy,
2. the LDS→VGPR read for each MFMA operand, and
3. the [pre-shuffled weight layout](../techniques/preshuffle-layout.md) that makes
   the global load contiguous and conflict-free.

Because these are composable layout objects, the compiler can fold `crd2idx`
chains at trace time and emit straight-line addressing, instead of you tracking
three independent index expressions by hand. The end-to-end worked GEMM —
including the pre-shuffle transform and a benchmark — is on the
[FlyDSL pre-shuffle GEMM kernel page](../kernels/flydsl-preshuffle-gemm.md).

## Beyond GEMM: flash attention and the MMA-atom API

FlyDSL is not GEMM-only. Its forward **flash attention** is a two-GEMM + online-softmax
kernel built the same way — and it now has two implementations behind one dispatcher: a
portable compiler-scheduled `flash_attn_generic.py` and a hand-scheduled, dual-wave
software-pipelined `flash_attn_gfx950.py` fast path (gfx950, `D=128`, bf16/f16). The full
walk-through — dispatch logic, the gfx950 hardware schedule, and the upstream PR arc — is
on the [FlyDSL flash-attention kernel page](../kernels/flydsl-flash-attention.md).

A recent direction is the **layout MMA-atom API**: instead of emitting raw ROCDL
intrinsics (`rocdl.mfma_f32_32x32x16_bf16`, `buffer_load_dwordx4`), kernels construct a
`make_mma_atom(...)` and issue it via `mma_atom_call_ssa(...)` (and `copy_atom_call_ssa`
for loads/stores). The flash-attention kernel was migrated onto this API, cutting ~1k
lines at parity — the same `Layout` algebra that describes tiling now also describes the
matrix-core operand distribution.

## Measured on MI350X (gfx950)

We profiled every major FlyDSL gfx950 kernel on real **MI350X silicon (ROCm 7.2)** with
rocprofv3 ATT + counters, against matched-shape AITER / CK / hipBLASLt baselines — see the
[profiling sweep & dashboard](../../sources/refs/ref-flydsl-kernel-profiling.md). Headlines
(throughput ÷ baseline, `>1` = FlyDSL faster):

- **Wins:** softmax **2.05×** (vs Triton), hgemm_splitk **1.66×**, moe_gemm **1.11×**.
- **Parity:** layernorm, quant, moe_reduce.
- **Headroom:** flash_attn 0.92×, mla 0.90×, rmsnorm 0.89×, paged-attention 0.48×, and the
  two big ones — **topk_gating 0.22×** and **rope 0.17×**.

Two recurring root causes, both actionable: (1) the attention/GEMM losers are
**register-pressure-capped** at 1–2 waves/SIMD (VGPR 175–251) — admit a 2nd wave by
[cutting the live set](../techniques/vgpr-budgeting.md); (2) rope/topk serialize cross-lane
reductions on `LGKMCNT` — replace with a DPP / `v_permlane16`
[wave reduction](../techniques/wave-reduce.md).

## Limitations and gotchas

- **Experimental / moving target.** Dialect op names, pass flags, and the Python
  API are not stable; pin a commit.
- **wave64 vs wave32.** CDNA targets (gfx942/gfx950/gfx1250) are wave64-only;
  RDNA4 (gfx1201) supports both. An atom chosen for an MFMA target is not valid
  on a WMMA target — query the arch, don't hardcode.
- **FP8 encoding differs by arch.** gfx942 FP8 is FNUZ; gfx950 FP8 is OCP. A
  layout/atom built for one is not bit-compatible with the other.
- **Debug at the ROCDL level.** When results are wrong, dump `--emit=rocdl` and
  confirm the matrix and `ds_*` intrinsics match what you intended before
  suspecting the algebra.

## See also

- [MFMA — AMD Matrix Core Instructions](../hardware/mfma.md)
- [Pre-shuffle layout technique](../techniques/preshuffle-layout.md)
- [FlyDSL pre-shuffle GEMM kernel](../kernels/flydsl-preshuffle-gemm.md)
- [Composable Kernel DSL](composable-kernel.md) · [Triton AMD backend](triton-amd.md)

## Sources

- [FlyDSL reference repository](https://github.com/ROCm/flydsl)
- [AITER — AMD AI operator library](https://github.com/ROCm/aiter)
- [LLVM AMDGPU backend — user guide & intrinsics](https://llvm.org/docs/AMDGPUUsage.html)
- [AMD Matrix Cores (programming overview)](https://gpuopen.com/learn/amd-lab-notes/amd-lab-notes-matrix-cores-readme/)
- [CDNA3 ISA Reference Guide](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf)
