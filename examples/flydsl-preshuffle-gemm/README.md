# FlyDSL Preshuffle GEMM — runnable example

This directory contains **two** artifacts illustrating the *preshuffle the
weights* idea from the FlyDSL `04-preshuffle_gemm.py` tutorial:

| File | What it is | Builds? | gfx950 runtime? |
|---|---|---|---|
| `04_preshuffle_gemm_flydsl.py` | Faithful FlyDSL reference snippet (layout transform + MFMA kernel) | — | **No** — reference only |
| `preshuffle_gemm_rocwmma.cpp` | rocWMMA API GEMM demonstrating the same idea | **Yes** | **Yes (gfx950)** |

## Why two files

The FlyDSL `.py` is a *pattern reference* and is not invoked by `build.sh`.
The runnable demonstration is the rocWMMA program. rocWMMA is the fragment API;
on gfx950 the compiler emits MFMA instructions. The program runs both kernels
and self-checks against a CPU reference.

## The idea being demonstrated

A wavefront matrix instruction does not consume a plain row-major tile — each
lane wants a specific 16×16 fragment of `B`. The demo builds two kernels:

1. **`gemm_rowmajor`** — loads `B` fragments straight out of the `K×N`
   row-major matrix with leading dim `ldb = N` (a strided fragment load every
   k-step).
2. **`gemm_preshuffled`** — `B` is permuted **once on the host**
   (`preshuffle_B`) so every 16×16 `(k,n)` tile is stored contiguously. The
   kernel then reads each fragment from a flat 16×16 block (`ldb = 16`,
   `base + offset`), with no per-tile address math. This mirrors FlyDSL's
   `copy(Bsh.tile(...), sB.next())` flat-copy path.

Both paths are checked against the same reference.

## Build & run

```bash
./build.sh            # defaults to gfx950 and runs the self-check
```

`build.sh` issues exactly:

```bash
hipcc --offload-arch=gfx950 -O3 -I/opt/rocm/include \
      preshuffle_gemm_rocwmma.cpp -o demo && ./demo
```

## Expected output (captured on MI355X / gfx950)

```
== Building rocWMMA preshuffle GEMM demo for gfx950 ==
== Running ==
Preshuffle GEMM demo (rocWMMA 16x16x16, fp16->fp32)  M=N=K=256
Correctness:
  row-major    max abs err = 0.0000  ->  PASS
  preshuffled  max abs err = 0.0000  ->  PASS
Timing (avg over 200 iters):
  row-major    0.0037 ms  (9013.2 GFLOP/s)
  preshuffled  0.0036 ms  (9396.8 GFLOP/s)

RESULT: PASS
```

The timing is one captured run; both numeric checks passed.

## Arch notes

- **Verified runtime:** MI355X / gfx950, where rocWMMA emits MFMA.
- **`warpSize`** is queried at runtime; one wave computes one 16×16 output tile.
- The FlyDSL `.py` snippet's MFMA path (`v_mfma_f32_16x16x16_f16`,
  AGPR accumulation) remains reference-only and is not exercised by `build.sh`.
