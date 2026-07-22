# Fused RMSNorm (portable HIP)

A self-contained, runnable RMSNorm forward kernel:

```
rms_i = sqrt( (1/H) * sum_j x_ij^2 + eps )
y_ij  = (x_ij / rms_i) * gamma_j      (gamma optional)
```

**One block per row.** Each row of `H` elements is processed by `BLOCK` lanes:

1. Each lane accumulates a strided partial `Σ x²` in **FP32** (even for FP16 IO —
   squaring in FP16 loses precision and biases the norm).
2. **Intra-wave** reduction with `__shfl_down` (lowers to `ds_bpermute`/DPP).
3. **Cross-wave** reduction through a tiny LDS array (one partial per wave),
   finalized by wave 0, then `rsqrt` broadcast via LDS.
4. Rescale and apply optional `gamma` in a second pass.

Nothing is hardcoded to a wave size: the kernel uses the `warpSize` builtin for
the shuffle stride and LDS partial count. The captured gfx950 run uses wave64.

## Classification

**PORTABLE** — pure HIP (FMA math, LDS, wave shuffles). No MFMA/WMMA. Builds and
runs on gfx950.

## Build & run

```bash
./build.sh                 # defaults to --offload-arch=gfx950, builds and runs
```

Or directly:

```bash
hipcc --offload-arch=gfx950 -O3 -std=c++17 rmsnorm.hip.cpp -o rmsnorm && ./rmsnorm
# gfx942 cross-compile only; no gfx942 runtime is claimed:
hipcc --offload-arch=gfx942 -O3 -std=c++17 -c rmsnorm.hip.cpp -o rmsnorm_gfx942.o
```

The program runs both an **fp32** IO path and an **fp16 IO / fp32 accumulate**
path, compares each against a CPU FP32 reference, and prints max-abs-error with
PASS/FAIL. Exit code is non-zero if any case fails.

## Expected output

Captured on AMD Instinct MI355X / gfx950:

```
Device: AMD Instinct MI355X  warpSize=64
---------------------------------------------------------------
rmsnorm fp32           [ 1024 x  4096] gamma=1  max|err|=2.384e-07  PASS  0.0099 ms  3373 GB/s
rmsnorm fp32 no-gamma  [  512 x  8192] gamma=0  max|err|=2.384e-07  PASS  0.0115 ms  2924 GB/s
rmsnorm fp32 odd-H     [  300 x  4097] gamma=1  max|err|=2.384e-07  PASS
rmsnorm fp16 IO        [ 1024 x  4096] gamma=1  max|err|=4.884e-04  PASS  0.0095 ms  1768 GB/s
rmsnorm fp16 IO big    [  256 x 16384] gamma=1  max|err|=4.884e-04  PASS  0.0145 ms  1160 GB/s
---------------------------------------------------------------
ALL TESTS PASSED
```

Tolerances: `1e-4` for fp32 IO, `3e-3` for fp16 IO (matches fp16's ~2⁻¹¹
storage resolution). The `odd-H` case (`H = 4097`) exercises the strided-tail
path where `H` is not a multiple of `BLOCK`.

The GB/s figures are a scalar-load reference point (1 read + 1 write), not a peak
result — this demo prioritizes correctness and portability over vectorization.
For peak bandwidth, switch to `float4`/`bf16x8` vectorized loads as described on
the wiki page.

## Notes

- Runs on **gfx950** (verified above). The same source cross-compiles for
  **gfx942**; this page does not claim a gfx942 runtime result.
- `MAX_WAVES = 32` bounds the LDS partials array for `BLOCK ≤ 1024` at any wave
  size ≥ 32.
