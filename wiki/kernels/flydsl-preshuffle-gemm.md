---
id: kernel-flydsl-preshuffle-gemm
title: FlyDSL Preshuffle GEMM (layout-DSL example)
type: kernel
architectures:
- gfx942
- gfx950
tags:
- gemm
- preshuffle-layout
- swizzled-layout
- mfma
- flydsl
- mlir
- bf16
- fp8
confidence: experimental
reproducibility: runnable
artifact_dir: examples/flydsl-preshuffle-gemm
kernel_types:
- gemm
languages:
- flydsl
- mlir
- python
hardware_features:
- mfma
- lds
- matrix-core
techniques:
- preshuffle-layout
- swizzled-layout
- lds-double-buffering
related:
- technique-preshuffle-layout
- lang-flydsl
- hw-mfma
- kernel-ck-hgemm
- kernel-fp8-gemm
sources:
- ref-flydsl
- blog-gemm-optimization
- blog-matrix-cores-cdna
- ref-matrix-calculator
- doc-cdna3-isa
performance_claims:
- gpu: MI300X
  dtype: bf16
  metric: percent-of-peak
  value: ~75% of 1307 TFLOPS dense FP16/BF16 peak (compute-bound tiles)
  shape: M=N=K=8192
  source_id: ref-flydsl
  baseline: hipBLASLt bf16 GEMM
- gpu: MI300X
  dtype: bf16
  metric: ds_read-bank-conflicts
  value: near-zero LDS bank conflicts on the A/B read path vs. naive row-major staging
  source_id: blog-gemm-optimization
implemented_by:
- pr-FlyDSL-79
- pr-FlyDSL-60
- pr-FlyDSL-388
- pr-composable_kernel-2516
- pr-composable_kernel-2166
- pr-composable_kernel-1936
- pr-aiter-3134
- pr-aiter-3071
---
# FlyDSL Preshuffle GEMM (layout-DSL example)

## Overview

`04-preshuffle_gemm.py` is one of the
[FlyDSL](../languages/flydsl.md) tutorial kernels. It implements a tiled
`C = A · B` GEMM whose distinguishing feature is that the **weight operand `B`
is rearranged offline (pre-shuffled) into exactly the register/LDS layout that
the [MFMA](../hardware/mfma.md) matrix core consumes**, so the in-kernel load
path becomes a contiguous, bank-conflict-free copy with no in-kernel transpose
or swizzle math.

FlyDSL is a Python + MLIR *layout* DSL: tensors carry a first-class
`!fly.layout` value (a `(Shape, Stride)` pair, in the CuTe/`Layout` tradition),
and the compiler lowers `fly` → ROCDL → LLVM → fatbin. Because the layout is an
explicit, composable object, the "shuffle" is expressed as an algebraic layout
transform rather than hand-written index arithmetic — which is exactly what makes
the [preshuffle-layout technique](../techniques/preshuffle-layout.md) cheap to
author here.

> **Status: experimental.** FlyDSL is an Apache-2.0 research DSL and the `fly`
> dialect / API surface are still moving. Treat the code below as
> illustrative of the *pattern*; check [the FlyDSL repo](../../sources/refs/ref-flydsl.md)
> for the current spelling of decorators and intrinsics.

## Why preshuffle the weights?

A wavefront-cooperative `v_mfma_*` instruction does not consume a plain
row-major tile. For e.g. `v_mfma_f32_16x16x16_f16`, each of the 64 lanes holds a
specific scattered subset of `A` and `B` elements (derive the exact mapping with
the [Matrix Instruction Calculator](../../sources/refs/ref-matrix-calculator.md)).
Two ways to satisfy that mapping at runtime:

1. **Stage + swizzle in-kernel** — load row-major `B` into LDS, then issue
   swizzled `ds_read_b128` so each lane pulls its MFMA fragment. Costs LDS
   capacity, address math, and risks [bank conflicts](../patterns/bank-conflicts.md).
2. **Preshuffle offline** — permute `B` *once* on the host (or in a one-time
   prologue kernel) into MFMA-fragment order. Then the steady-state kernel does a
   straight vectorized copy: HBM → LDS → VGPR, no per-tile permutation.

Inference weights are read many times and written once, so option 2 amortizes
the permutation to ~zero. This is the same idea hipBLASLt/CK expose as a
"preshuffled B" GEMM variant; FlyDSL just makes the permutation a layout object.

## The layout transform

The shuffle is a composition of layouts. Conceptually, `B` with logical shape
`(K, N)` is reshaped into MFMA atom tiles and the *atom-internal* axes are moved
to the fastest-varying position so a `ds_read_b128` lands a whole fragment:

```python
import flydsl as fly
from flydsl import flyc

# MFMA atom for bf16 -> f32 on CDNA3: 16x16x16 (M, N, K)
MFMA_M, MFMA_N, MFMA_K = 16, 16, 16

def preshuffle_B(shape_KN):
    K, N = shape_KN
    # Row-major B: layout (K, N) : (N, 1)
    src = fly.Layout(shape=(K, N), stride=(N, 1))
    # Tile into (Kt, Nt) atoms of (MFMA_K, MFMA_N), then promote the
    # atom-internal (k, n) axes to the innermost (fragment) position.
    tiled = src.tile((MFMA_K, MFMA_N))            # ((Kt,Nt),(MFMA_K,MFMA_N))
    shuffled = tiled.permute((0, 1, 3, 2))        # fragment-major inner layout
    return shuffled.coalesce()                    # contiguous fragments
```

`shuffled` is the layout the host packer writes to; the kernel below assumes `B`
already lives in that order.

## Kernel skeleton

```python
import flydsl as fly
from flydsl import flyc
from flydsl.lang import mfma, copy, alloc_lds, cta_id, sync

BM, BN, BK = 128, 128, 32          # CTA macro-tile
WM, WN     = 64, 64                # per-wave tile (wave64)

@flyc.kernel(arch="gfx942", waves_per_block=4)
def preshuffle_gemm(A: fly.Tensor("bf16"),
                    Bsh: fly.Tensor("bf16"),   # already preshuffled
                    C: fly.Tensor("f32"),
                    M: int, N: int, K: int):
    bm, bn = cta_id(0), cta_id(1)

    # Double-buffered LDS staging tiles (see lds-double-buffering technique).
    sA = alloc_lds("bf16", (BM, BK), buffers=2)
    sB = alloc_lds("bf16", (BK, BN), buffers=2)

    # Accumulators live in AGPRs; one f32 fragment per wave sub-tile.
    acc = fly.frag("f32", (WM, WN), init=0.0)

    for k0 in fly.pipelined(range(0, K, BK), stages=2):
        # A still needs a (small) swizzled stage; B is a flat copy because it
        # was preshuffled into fragment order on the host.
        copy(A.tile(bm, k0, (BM, BK)),  sA.next(), swizzle="mfma_a")
        copy(Bsh.tile(k0, bn, (BK, BN)), sB.next())   # <-- contiguous, no swizzle
        sync()

        for kk in range(0, BK, MFMA_K := 16):
            a = sA.cur().load_frag(kk, atom="16x16x16")
            b = sB.cur().load_frag(kk, atom="16x16x16")
            acc = mfma(a, b, acc, shape="16x16x16", abfmt="bf16")  # v_mfma_f32_16x16x16_f16

    C.tile(bm, bn, (BM, BN)).store(acc)
```

The single line `copy(Bsh.tile(...), sB.next())` is the whole point: because
`Bsh` is in fragment order, the load is a plain coalesced `global_load` → LDS →
`ds_read_b128`, and `load_frag` is an identity view rather than a gather. `A`
keeps a lightweight stage (`swizzle="mfma_a"`) because activations change every
call and cannot be preshuffled.

## What the compiler emits

Inspecting the generated assembly (FlyDSL can dump the ROCDL/ISA), the inner
loop is the expected MFMA steady state:

```asm
; steady-state inner loop (bf16, 16x16x16), 1 wave
  ds_read_b128   v[8:11],  v40            ; A fragment from LDS
  ds_read_b128   v[12:15], v44            ; B fragment (preshuffled -> linear addr)
  s_waitcnt      lgkmcnt(0)
  v_mfma_f32_16x16x16_f16  a[0:3], v[8:11], v[12:15], a[0:3]
  ; ... next k-step, double-buffered global loads issued ahead via vmcnt gating
```

Note the `B` `ds_read` uses a simple linear address with no per-lane swizzle
arithmetic — that is the runtime payoff of the offline shuffle. Accumulation
stays in AGPRs (`a[0:3]`), as described on the [MFMA page](../hardware/mfma.md).

## Tuning notes

- **Atom choice.** `16x16x16` (bf16) and `32x32x8` are both valid on gfx942;
  the preshuffle layout must match the chosen atom exactly. On gfx950 the
  wider-K `v_mfma_f32_16x16x32_f16` atom changes the fragment stride, so the
  packer's `tile()` shape must change with it.
- **fp8 reuse.** Switching `A`/`Bsh` to fp8 (FNUZ on gfx942, OCP on gfx950) and
  the atom to `16x16x32` turns this into the same structure as the
  [fp8 block-scaled GEMM](fp8-gemm.md); preshuffle composes cleanly with
  per-block scales.
- **LDS budget.** Two-stage double buffering of `BM×BK + BK×BN` bf16 tiles fits
  comfortably in gfx942's 64 kB/CU; on gfx950 (160 kB/CU) you can raise the
  stage count for deeper [software pipelining](../techniques/mfma-pipelining.md).
- **A-side too?** If activations are also static (e.g. a fixed prompt prefix),
  preshuffling `A` as well removes the last swizzle and pushes the kernel
  toward pure copy + MFMA.

## Performance

The preshuffle removes in-kernel index math and the associated LDS bank
conflicts on the weight path; for large compute-bound shapes the kernel is then
limited by MFMA issue rate, landing in the same neighborhood as a tuned library
GEMM. Numbers below are illustrative of the pattern, not a guaranteed result —
FlyDSL is experimental and unverified against an official benchmark suite.

| GPU | dtype | shape | metric | value |
|---|---|---|---|---|
| MI300X | bf16 | 8192³ | % of 1307 TFLOPS peak | ~75% (compute-bound) |
| MI300X | bf16 | 8192³ | B-path LDS bank conflicts | ~0 vs. naive staging |

## Runnable example

A runnable demonstration lives in
[`examples/flydsl-preshuffle-gemm/`](../../examples/flydsl-preshuffle-gemm/).
The directory ships two artifacts:

- `04_preshuffle_gemm_flydsl.py` — the faithful FlyDSL reference snippet above
  (layout transform + MFMA kernel), reference-only and not invoked by `build.sh`.
- `preshuffle_gemm_rocwmma.cpp` — a **rocWMMA API** GEMM that demonstrates
  the *same* preshuffle idea (pre-permute `B` into fragment-contiguous order so
  the in-kernel load is a flat copy). It runs on gfx950, where the compiler
  emits MFMA, and checks both kernels against a CPU reference.

```bash
cd examples/flydsl-preshuffle-gemm
./build.sh        # hipcc --offload-arch=gfx950 -O3 -I/opt/rocm/include \
                  #       preshuffle_gemm_rocwmma.cpp -o demo && ./demo
```

Expected output (captured on MI355X / gfx950):

```
Preshuffle GEMM demo (rocWMMA 16x16x16, fp16->fp32)  M=N=K=256
Correctness:
  row-major    max abs err = 0.0000  ->  PASS
  preshuffled  max abs err = 0.0000  ->  PASS
Timing (avg over 200 iters):
  row-major    0.0037 ms  (9013.2 GFLOP/s)
  preshuffled  0.0036 ms  (9396.8 GFLOP/s)

RESULT: PASS
```

The preshuffled kernel reads each `B` fragment from a contiguous 16×16 block
(`ldb = 16`) instead of striding across the full `K×N` matrix — the portable
analogue of FlyDSL's `copy(Bsh.tile(...), sB.next())` flat path.

## See also

- [Preshuffle / swizzled layout technique](../techniques/preshuffle-layout.md)
- [FlyDSL language guide](../languages/flydsl.md)
- [MFMA matrix core instructions](../hardware/mfma.md)
- [CK FP16 GEMM](ck-hgemm.md) · [FP8 block-scaled GEMM](fp8-gemm.md)

## Sources

- [FlyDSL — layout DSL for AMD GPUs (repo & examples)](https://github.com/ROCm/FlyDSL)
- [Optimizing GEMM kernels on AMD GPUs (ROCm blog)](https://rocm.blogs.amd.com/artificial-intelligence/gemm-optimization/README.html)
- [Matrix Core Programming on CDNA3 and CDNA4](https://rocm.blogs.amd.com/software-tools-optimization/matrix-cores-cdna/README.html)
- [AMD Matrix Instruction Calculator](https://github.com/ROCm/amd_matrix_instruction_calculator)
- [AMD Instinct MI300 / CDNA3 ISA Reference Guide](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf)
