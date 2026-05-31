# FP8 Block-Scaled GEMM — runnable example

This directory contains **two** GEMM examples that together demonstrate the FP8
GEMM page. They are deliberately separate because the FP8 matrix-core path is a
**CDNA-only (MFMA)** instruction set, while the box this was authored on is an
**RX 9070 XT = gfx1201 = RDNA4**, which has **WMMA, not MFMA**.

| File | What | Arch | Runs here? |
|---|---|---|---|
| `wmma_hgemm.cpp` | Portable rocWMMA **FP16** tiled GEMM (16×16×16), CPU-checked | gfx1201 (and any rocWMMA target) | ✅ builds **and runs** |
| `fp8_gemm_cdna.cpp` | CDNA-MFMA **FP8** GEMM via the real `f8f6f4` / `fp8_fp8` builtins | gfx950 (CDNA4) + gfx942 (CDNA3) | ❌ cross-compile-verify only |

## Build / run everything

```bash
./build.sh
```

`build.sh` exits 0 only if: the portable kernel builds **and self-checks PASS**,
the gfx950 object+exe build, the gfx942 object builds, and the expected
matrix-core instructions are confirmed present in the emitted device assembly.

---

## 1. Portable rocWMMA FP16 GEMM (runs on gfx1201)

The demonstrable fallback. Same tiled matrix-core structure as the FP8 kernel,
but in FP16 using rocWMMA, which abstracts WMMA on RDNA and MFMA on CDNA so it
runs natively here. Each wave computes one 16×16 output tile, FP32 accumulate,
result checked against a CPU reference.

```bash
hipcc --offload-arch=gfx1201 -I/opt/rocm/include wmma_hgemm.cpp -o wmma_hgemm
./wmma_hgemm
```

**Real captured output (gfx1201, ROCm 7.2.3):**

```
rocWMMA FP16 GEMM  M=256 N=256 K=256 (warpSize=32)
avg 0.0069 ms/iter   4863.5 GFLOP/s
max abs error = 0.000000
PASS
```

(Timing is a tiny 256³ tile — illustrative, not a peak-throughput benchmark.)

---

## 2. CDNA-MFMA FP8 GEMM (cross-compile-verify only)

This is the kernel the wiki page is actually about. It calls the real
matrix-core builtins and is verified to **compile and emit the right
instructions** for CDNA targets. It will **not execute on gfx1201** (no MFMA),
so it is not run here — run it on MI350X/MI355X (gfx950) or MI300X (gfx942).

```bash
# gfx950 (CDNA4): OCP E4M3, unified f8f6f4 scaled MMA, K=128
hipcc --offload-arch=gfx950 -c fp8_gemm_cdna.cpp -o fp8_gemm_gfx950.o
hipcc --offload-arch=gfx950    fp8_gemm_cdna.cpp -o fp8_gemm_gfx950

# gfx942 (CDNA3): FNUZ E4M3, K=32 fp8_fp8 MMA, software block scaling
hipcc --offload-arch=gfx942 -c fp8_gemm_cdna.cpp -o fp8_gemm_gfx942.o
```

Confirmed in the emitted device assembly (`build.sh` greps for these):

- gfx950 → `v_mfma_scale_f32_16x16x128_f8f6f4`
- gfx942 → `v_mfma_f32_16x16x32_fp8_fp8`

The toolchain exposes the **scaled** f8f6f4 builtin
(`__builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4`); passing scale operands of
`0` gives a plain dense GEMM, while real per-32-K-block **E8M0** exponents give
MXFP8 with the scale applied *in hardware* during accumulation.

### FNUZ vs OCP caveat (the important one)

- **gfx950 / CDNA4 → OCP FP8** (E4M3/E5M2 per the OCP spec): has signed zero and
  Inf/NaN, and a **K=128** unified `f8f6f4` MMA with **hardware MX block
  scaling** (E8M0 scale per 32 elements).
- **gfx942 / CDNA3 → FNUZ FP8** (`__hip_fp8_e4m3_fnuz`): *Finite, Unsigned-zero,
  NaN-only* — no Inf, no negative zero, one extra usable exponent code. The MMA
  is **K=32** and there is **no hardware MX**, so block scales are dequantized in
  software (see the per-block multiply in `fp8_gemm_gfx942`).

Because the bit encodings differ, **FP8 weights are not bit-compatible across the
two** — re-encode when porting gfx942 ↔ gfx950. The kernel layouts here are
minimal (no LDS staging / double buffering / 4-wave schedule) and exist to prove
the instruction path compiles; the production fast kernel uses Composable Kernel
or hipBLASLt.

## Arch summary

- **Runs on:** gfx1201 (the rocWMMA FP16 demo).
- **Cross-compiles for / runs on:** gfx950 (MI350X/MI355X) and gfx942 (MI300X)
  for the FP8 MFMA kernel — built and instruction-verified here, executed on
  CDNA hardware.
