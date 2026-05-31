// vadd_hip.cpp — PORTABLE HIP vector add: C[i] = A[i] + B[i] (FP32).
// Runs natively on gfx1201 (RDNA4) and on CDNA. Grid-stride loop so a fixed
// launch handles any N. Self-checks against a CPU reference and prints PASS/FAIL
// plus an effective-bandwidth estimate (12 B/elem: 2 reads + 1 write).
//
// This is the "what you'd usually ship" companion to the hand-written GCN
// assembly variant (see vadd_asm_gfx942.cpp) on the wiki page.

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

#define HIP_CHECK(cmd)                                                          \
  do {                                                                          \
    hipError_t e = (cmd);                                                       \
    if (e != hipSuccess) {                                                      \
      std::fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(e),     \
                   __FILE__, __LINE__);                                         \
      std::exit(1);                                                             \
    }                                                                           \
  } while (0)

// Grid-stride persistent-style vector add. Each thread walks the array in
// strides of gridDim.x*blockDim.x, so launch/teardown is amortized for large N.
__global__ void vadd(const float* __restrict__ A, const float* __restrict__ B,
                     float* __restrict__ C, int N) {
  int stride = gridDim.x * blockDim.x;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += stride) {
    C[i] = A[i] + B[i];
  }
}

int main() {
  const int N = 1 << 24;  // 16M elements
  const size_t bytes = size_t(N) * sizeof(float);

  std::vector<float> hA(N), hB(N), hC(N), ref(N);
  for (int i = 0; i < N; ++i) {
    hA[i] = float((i % 1000) * 0.5f);
    hB[i] = float((i % 777) * 0.25f) - 13.0f;
    ref[i] = hA[i] + hB[i];
  }

  float *dA, *dB, *dC;
  HIP_CHECK(hipMalloc(&dA, bytes));
  HIP_CHECK(hipMalloc(&dB, bytes));
  HIP_CHECK(hipMalloc(&dC, bytes));
  HIP_CHECK(hipMemcpy(dA, hA.data(), bytes, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(dB, hB.data(), bytes, hipMemcpyHostToDevice));

  const int block = 256;
  // Persistent-ish launch: a couple of waves' worth per CU, capped so each
  // thread processes many elements via the grid-stride loop.
  const int grid = 4096;

  // Warm-up.
  hipLaunchKernelGGL(vadd, dim3(grid), dim3(block), 0, 0, dA, dB, dC, N);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());

  // Timed runs.
  hipEvent_t t0, t1;
  HIP_CHECK(hipEventCreate(&t0));
  HIP_CHECK(hipEventCreate(&t1));
  const int iters = 50;
  HIP_CHECK(hipEventRecord(t0));
  for (int it = 0; it < iters; ++it)
    hipLaunchKernelGGL(vadd, dim3(grid), dim3(block), 0, 0, dA, dB, dC, N);
  HIP_CHECK(hipEventRecord(t1));
  HIP_CHECK(hipEventSynchronize(t1));
  float ms = 0.0f;
  HIP_CHECK(hipEventElapsedTime(&ms, t0, t1));
  ms /= iters;

  HIP_CHECK(hipMemcpy(hC.data(), dC, bytes, hipMemcpyDeviceToHost));

  // CPU reference self-check.
  double max_abs_err = 0.0;
  for (int i = 0; i < N; ++i) {
    double e = std::fabs(double(hC[i]) - double(ref[i]));
    if (e > max_abs_err) max_abs_err = e;
  }

  double gbytes = 3.0 * double(bytes) / 1e9;  // 2 reads + 1 write
  double bw = gbytes / (ms / 1e3);

  bool pass = (max_abs_err == 0.0);  // exact for this integer-derived data
  std::printf("vadd HIP (portable, gfx1201): N=%d  block=%d grid=%d\n", N, block,
              grid);
  std::printf("  time = %.3f ms/iter   effective BW = %.1f GB/s (12 B/elem)\n",
              ms, bw);
  std::printf("  max abs err = %.3g\n", max_abs_err);
  std::printf("  %s\n", pass ? "PASS" : "FAIL");

  HIP_CHECK(hipFree(dA));
  HIP_CHECK(hipFree(dB));
  HIP_CHECK(hipFree(dC));
  HIP_CHECK(hipEventDestroy(t0));
  HIP_CHECK(hipEventDestroy(t1));
  return pass ? 0 : 1;
}
