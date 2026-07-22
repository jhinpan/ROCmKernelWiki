// LDS-staged, bank-conflict-free out-of-place matrix transpose.
// Portable pure HIP — builds and runs on gfx950; the guarded source also
// device-compiles for gfx942.
//
// Pattern: load a TILE x TILE block with a coalesced row-major read into LDS,
// then write it out transposed with another coalesced row-major write. The
// transposed read walks down an LDS column, which is exactly where bank
// conflicts appear. For this block(32,32) lane mapping the required padding is
// target-specific: +1 dword on gfx942, +2 on gfx950 where one b32 phase contains
// two adjacent columns across the same 32 rows.
//
// Self-checks the result against a CPU reference and reports effective
// bandwidth.

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cmath>

#define TILE 32
#if defined(__gfx950__)
#define LDS_PAD 2
#else
#define LDS_PAD 1
#endif

#define HIP_CHECK(cmd)                                                          \
  do {                                                                          \
    hipError_t e = (cmd);                                                       \
    if (e != hipSuccess) {                                                      \
      fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(e),         \
              __FILE__, __LINE__);                                              \
      exit(1);                                                                  \
    }                                                                           \
  } while (0)

// out[x][y] = in[y][x]; in is rows x cols (row-major), out is cols x rows.
__global__ void transpose_lds(float* __restrict__ out,
                              const float* __restrict__ in,
                              int rows, int cols) {
  __shared__ float tile[TILE][TILE + LDS_PAD];

  int x = blockIdx.x * TILE + threadIdx.x;  // input column
  int y = blockIdx.y * TILE + threadIdx.y;  // input row

  // Coalesced read: consecutive threadIdx.x -> consecutive global columns.
  if (x < cols && y < rows)
    tile[threadIdx.y][threadIdx.x] = in[(size_t)y * cols + x];

  __syncthreads();

  // Output tile is the transposed block: swap block coordinates.
  int xo = blockIdx.y * TILE + threadIdx.x;  // output column (= input row)
  int yo = blockIdx.x * TILE + threadIdx.y;  // output row    (= input col)

  // Transposed LDS read (down a column) + coalesced global write.
  if (xo < rows && yo < cols)
    out[(size_t)yo * rows + xo] = tile[threadIdx.x][threadIdx.y];
}

int main(int argc, char** argv) {
  int rows = (argc > 1) ? atoi(argv[1]) : 2048;
  int cols = (argc > 2) ? atoi(argv[2]) : 4096;

  printf("Transpose %d x %d (fp32), TILE=%d\n", rows, cols, TILE);

  size_t n = (size_t)rows * cols;
  size_t bytes = n * sizeof(float);

  std::vector<float> h_in(n), h_out(n);
  for (size_t i = 0; i < n; ++i) h_in[i] = (float)((i * 1315423911u) % 10007) * 0.5f;

  float *d_in, *d_out;
  HIP_CHECK(hipMalloc(&d_in, bytes));
  HIP_CHECK(hipMalloc(&d_out, bytes));
  HIP_CHECK(hipMemcpy(d_in, h_in.data(), bytes, hipMemcpyHostToDevice));

  dim3 block(TILE, TILE);
  dim3 grid((cols + TILE - 1) / TILE, (rows + TILE - 1) / TILE);

  // Warmup.
  hipLaunchKernelGGL(transpose_lds, grid, block, 0, 0, d_out, d_in, rows, cols);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());

  // Timed runs.
  const int iters = 50;
  hipEvent_t start, stop;
  HIP_CHECK(hipEventCreate(&start));
  HIP_CHECK(hipEventCreate(&stop));
  HIP_CHECK(hipEventRecord(start));
  for (int i = 0; i < iters; ++i)
    hipLaunchKernelGGL(transpose_lds, grid, block, 0, 0, d_out, d_in, rows, cols);
  HIP_CHECK(hipEventRecord(stop));
  HIP_CHECK(hipEventSynchronize(stop));
  float ms = 0.f;
  HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
  float avg_ms = ms / iters;
  // read + write of the whole matrix per pass.
  double gbps = (2.0 * bytes) / (avg_ms * 1e-3) / 1e9;

  HIP_CHECK(hipMemcpy(h_out.data(), d_out, bytes, hipMemcpyDeviceToHost));

  // CPU reference check: out[c*rows + r] == in[r*cols + c].
  double max_abs_err = 0.0;
  size_t bad = 0;
  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < cols; ++c) {
      float expect = h_in[(size_t)r * cols + c];
      float got = h_out[(size_t)c * rows + r];
      double e = fabs((double)got - (double)expect);
      if (e > max_abs_err) max_abs_err = e;
      if (e != 0.0 && bad < 5)
        fprintf(stderr, "mismatch at (%d,%d): got %f expect %f\n", r, c, got, expect);
      if (e != 0.0) ++bad;
    }
  }

  printf("avg kernel time: %.3f ms   effective BW: %.1f GB/s\n", avg_ms, gbps);
  printf("max abs error: %g   mismatches: %zu\n", max_abs_err, bad);

  bool pass = (max_abs_err == 0.0);  // transpose is a pure copy: must be exact.
  printf("%s\n", pass ? "PASS" : "FAIL");

  HIP_CHECK(hipEventDestroy(start));
  HIP_CHECK(hipEventDestroy(stop));
  HIP_CHECK(hipFree(d_in));
  HIP_CHECK(hipFree(d_out));
  return pass ? 0 : 1;
}
