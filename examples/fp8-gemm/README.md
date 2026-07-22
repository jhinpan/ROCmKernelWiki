# FP8 Block-Scaled GEMM — ISA probe + runtime fallback

This directory contains two deliberately separate paths: an FP8 compiler/ISA
probe and a runnable FP16 correctness fallback.

| File | What | Target | Runtime status |
|---|---|---|---|
| `wmma_hgemm.cpp` | rocWMMA API **FP16** tiled GEMM, CPU-checked | gfx950 | Builds, runs, passes |
| `fp8_gemm_cdna.cpp` | **FP8** builtins for `f8f6f4` / `fp8_fp8` | gfx950 + gfx942 | Compiler/ISA-only; no kernel launch |

## Build and verify

```bash
./build.sh
```

`build.sh` exits zero only if the FP16 fallback self-check passes, the FP8
gfx950 object/executable and gfx942 object build, and both expected instructions
are present in emitted device assembly.

---

## 1. rocWMMA API FP16 GEMM (runs on gfx950)

The runnable fallback uses the same tiled matrix-core structure as the FP8
kernel, but with FP16 inputs through rocWMMA. rocWMMA is the API; gfx950 emits
MFMA instructions. Each wave computes one 16×16 output tile with FP32
accumulation, checked against a CPU reference.

```bash
hipcc --offload-arch=gfx950 -I/opt/rocm/include wmma_hgemm.cpp -o wmma_hgemm
./wmma_hgemm
```

**Captured output (MI355X / gfx950):**

```
rocWMMA FP16 GEMM  M=256 N=256 K=256 (warpSize=64)
avg 0.0035 ms/iter   9657.6 GFLOP/s
max abs error = 0.000000
PASS
```

(Timing is a tiny 256³ tile — illustrative, not a peak-throughput benchmark.)

---

## 2. FP8 GEMM (compiler/ISA verification only)

This is the kernel the wiki page is actually about. It calls the real
matrix-core builtins and is verified to compile and emit the expected
instructions. `build.sh` does not execute the FP8 binary, and the linked host
`main()` does not launch a kernel. This is not a numeric FP8 correctness test.

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

- **gfx950 runtime:** the rocWMMA FP16 fallback runs and passes on MI355X.
- **gfx950 FP8:** compiler/link/ISA verification only; no kernel launch.
- **gfx942 FP8:** object cross-compile and ISA verification only; no runtime is
  claimed.
