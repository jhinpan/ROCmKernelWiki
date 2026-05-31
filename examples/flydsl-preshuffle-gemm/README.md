# FlyDSL Preshuffle GEMM — runnable example

This directory contains **two** artifacts illustrating the *preshuffle the
weights* idea from the FlyDSL `04-preshuffle_gemm.py` tutorial:

| File | What it is | Builds? | Runs here? |
|---|---|---|---|
| `04_preshuffle_gemm_flydsl.py` | Faithful FlyDSL reference snippet (layout transform + MFMA kernel) | — | **No** — reference only |
| `preshuffle_gemm_rocwmma.cpp` | Portable rocWMMA HIP GEMM demonstrating the same idea | **Yes** | **Yes (gfx1201)** |

## Why two files

FlyDSL is **not installed** on this box and it targets **CDNA MFMA**
(gfx942/gfx950). So the `.py` is included verbatim as a *pattern reference*
(it raises immediately if executed). The runnable demonstration is the
**portable rocWMMA** program: rocWMMA abstracts the wave matrix instruction
(WMMA on gfx1201 / MFMA on CDNA), so it compiles and runs natively here and
self-checks against a CPU reference.

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

Both produce identical numerics; the preshuffled layout is the runtime payoff.

## Build & run

```bash
./build.sh            # builds for gfx1201 and runs the self-check
# or pick an arch explicitly:
./build.sh gfx1201
```

`build.sh` issues exactly:

```bash
hipcc --offload-arch=gfx1201 -O3 -I/opt/rocm/include \
      preshuffle_gemm_rocwmma.cpp -o demo && ./demo
```

## Expected output (captured on this gfx1201 / RX 9070 XT, ROCm 7.2.3, rocWMMA 2.2.0)

```
== Building rocWMMA preshuffle GEMM demo for gfx1201 ==
== Running ==
Preshuffle GEMM demo (rocWMMA 16x16x16, fp16->fp32)  M=N=K=256
Correctness:
  row-major    max abs err = 0.0000  ->  PASS
  preshuffled  max abs err = 0.0000  ->  PASS
Timing (avg over 200 iters):
  row-major    0.0102 ms  (3302.1 GFLOP/s)
  preshuffled  0.0064 ms  (5254.8 GFLOP/s)

RESULT: PASS
```

(Exact GFLOP/s vary run to run; both kernels always PASS the numeric check.
The preshuffled layout is consistently faster because its `B` fragment loads
are contiguous instead of strided.)

## Arch notes

- **Runs on:** gfx1201 (RDNA4, WMMA) — verified above. Portable rocWMMA also
  runs on CDNA (gfx942/gfx950, MFMA) unchanged.
- **`warpSize`** is queried at runtime (32 on gfx1201, 64 on CDNA); one wave
  computes one 16×16 output tile.
- The FlyDSL `.py` snippet's MFMA path (`v_mfma_f32_16x16x16_f16`,
  AGPR accumulation) is what FlyDSL would emit on **gfx942/gfx950** — it is not
  exercised here.
