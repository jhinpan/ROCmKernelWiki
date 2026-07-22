# ck-hgemm — portable rocWMMA FP16 GEMM (16×16×16)

A runnable, self-checking FP16 GEMM built on **rocWMMA** fragments. Each wave
computes one 16×16 output tile, accumulating over K in 16×16×16 fragments with an
FP32 accumulator. It verifies against a CPU reference (fp16 inputs, fp32 accumulate)
and prints `PASS`/`FAIL` plus a rough TFLOPS number.

This is the **portable, demonstrable companion** to the Composable Kernel (CK)
hgemm page. CK/ck_tile is the production GEMM route; this example uses the
rocWMMA API and emits MFMA instructions on gfx950.

## What it shows

- rocWMMA `fragment` / `load_matrix_sync` / `mma_sync` / `store_matrix_sync` for a
  16×16×16 FP16 tile, accumulating across K.
- Layout: `A` MxK row-major, `B` KxN col-major (`matrix_b` `col_major`), `C` MxN
  row-major FP32. One wave per 16×16 tile, 4×4 waves per block.
- Numeric self-check vs CPU sgemm with a K-scaled tolerance.

## Arch: runs vs cross-compiles

- **Verified runtime:** MI355X / gfx950. `mma_sync` is a rocWMMA API call; the
  gfx950 device code uses **MFMA**.
- **gfx942:** the same source can be cross-compiled with
  `--offload-arch=gfx942`; no gfx942 runtime is claimed.
- **Production route on CDNA:** for peak FP16 GEMM throughput on MI300X/MI350 use
  **Composable Kernel / ck_tile** (block→warp→MFMA tiling, LDS double buffering,
  MFMA software pipelining) rather than a single-tile-per-wave rocWMMA kernel.
  See the parent wiki page. This example is for clarity and on-box verification,
  not as a peak-performance kernel.

## Build & run

```bash
./build.sh
# or directly:
hipcc --offload-arch=gfx950 -O3 -std=c++17 -I/opt/rocm/include \
      hgemm_wmma.cpp -o hgemm_wmma
./hgemm_wmma 256 256 256      # M N K (rounded up to multiples of 16)
```

Cross-compile the same source for CDNA (object only, not run here):

```bash
hipcc --offload-arch=gfx942 -O3 -std=c++17 -I/opt/rocm/include -c hgemm_wmma.cpp -o hgemm_gfx942.o
```

## Expected output (captured on MI355X / gfx950)

```
rocWMMA FP16 GEMM  M=256 N=256 K=256  (16x16x16 fragments)
Device: AMD Instinct MI355X (gfx950:sramecc+:xnack-), warpSize=64
max abs err = 0.0000   max rel err = 0.01253   (tol abs = 10.24)
avg kernel time = 0.0156 ms   ~2150.4 GFLOP/s (2.150 TFLOPS)
PASS
```

The TFLOPS figure is from one small-shape run, not a tuned peak.
