# ck-hgemm — portable rocWMMA FP16 GEMM (16×16×16)

A runnable, self-checking FP16 GEMM built on **rocWMMA** fragments. Each wave
computes one 16×16 output tile, accumulating over K in 16×16×16 fragments with an
FP32 accumulator. It verifies against a CPU reference (fp16 inputs, fp32 accumulate)
and prints `PASS`/`FAIL` plus a rough TFLOPS number.

This is the **portable, demonstrable companion** to the Composable Kernel (CK)
hgemm page. The CK/ck_tile path is the production GEMM route on CDNA; this rocWMMA
example shows the same matrix-core abstraction running natively on this RDNA4 box.

## What it shows

- rocWMMA `fragment` / `load_matrix_sync` / `mma_sync` / `store_matrix_sync` for a
  16×16×16 FP16 tile, accumulating across K.
- Layout: `A` MxK row-major, `B` KxN col-major (`matrix_b` `col_major`), `C` MxN
  row-major FP32. One wave per 16×16 tile, 4×4 waves per block.
- Numeric self-check vs CPU sgemm with a K-scaled tolerance.

## Arch: runs vs cross-compiles

- **Runs here:** gfx1201 (RDNA4, RX 9070 XT). rocWMMA lowers `mma_sync` to the
  RDNA **WMMA** instruction and runs natively. Verified — see output below.
- **Same source on CDNA:** on gfx942 / gfx950 rocWMMA lowers the identical
  `mma_sync` to **MFMA** (`v_mfma_f32_16x16x16_f16`). The source is portable; build
  with `--offload-arch=gfx942` / `gfx950`.
- **Production route on CDNA:** for peak FP16 GEMM throughput on MI300X/MI350 use
  **Composable Kernel / ck_tile** (block→warp→MFMA tiling, LDS double buffering,
  MFMA software pipelining) rather than a single-tile-per-wave rocWMMA kernel.
  See the parent wiki page. This example is for clarity and on-box verification,
  not as a peak-performance kernel.

## Build & run

```bash
./build.sh
# or directly:
hipcc --offload-arch=gfx1201 -O3 -std=c++17 -I/opt/rocm/include \
      hgemm_wmma.cpp -o hgemm_wmma
./hgemm_wmma 256 256 256      # M N K (rounded up to multiples of 16)
```

Cross-compile the same source for CDNA (object only, not run here):

```bash
hipcc --offload-arch=gfx942 -O3 -std=c++17 -I/opt/rocm/include -c hgemm_wmma.cpp -o hgemm_gfx942.o
```

## Expected output (real, captured on gfx1201)

```
rocWMMA FP16 GEMM  M=256 N=256 K=256  (16x16x16 fragments)
Device: AMD Radeon RX 9070 XT (gfx1201), warpSize=32
max abs err = 0.0000   max rel err = 0.00783   (tol abs = 10.24)
avg kernel time = 0.0081 ms   ~4120.5 GFLOP/s (4.121 TFLOPS)
PASS
```

The TFLOPS figure is a small-shape illustrative number from this single-tile-per-wave
kernel, not a tuned peak — the CK path is what reaches ~1000+ TFLOPS on MI300X.
