// Portable fused RMSNorm — one block per row.
//
// rms_i  = sqrt( (1/H) * sum_j x_ij^2 + eps )
// y_ij   = (x_ij / rms_i) * gamma_j        (gamma optional)
//
// Reduction: intra-wave with __shfl_down, cross-wave through a small LDS array.
// Accumulation is always FP32 even for FP16 I/O (squaring FP16 loses precision).
//
// Builds and runs on gfx950. warpSize is queried at runtime via the host and the
// kernel uses __shfl_down plus a wave-count-agnostic LDS finalize, so the wave
// width is not hardcoded.
//
// Self-checks fp32 and fp16-IO paths against a CPU reference and prints PASS/FAIL.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <random>

#define HIP_CHECK(cmd)                                                          \
  do {                                                                         \
    hipError_t e = (cmd);                                                      \
    if (e != hipSuccess) {                                                     \
      fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(e),         \
              __FILE__, __LINE__);                                             \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

// Max waves we ever finalize across (BLOCK<=1024, wave>=32 -> <=32 partials).
static constexpr int MAX_WAVES = 32;

// One block per row. Type IO is the storage dtype (float or __half);
// all reduction/scale math is FP32.
template <typename IO, int BLOCK>
__global__ void rmsnorm_kernel(IO* __restrict__ y, const IO* __restrict__ x,
                               const IO* __restrict__ gamma, float eps, int H) {
  const int row  = blockIdx.x;
  const int tid  = threadIdx.x;
  const IO* xr   = x + (size_t)row * H;
  IO*       yr   = y + (size_t)row * H;

  // 1) Partial sum of squares in FP32.
  float local = 0.f;
  for (int i = tid; i < H; i += BLOCK) {
    float v = static_cast<float>(xr[i]);
    local += v * v;
  }

  // 2) Intra-wave reduction. warpSize is a device builtin.
  const int ws = warpSize;
  for (int off = ws >> 1; off > 0; off >>= 1)
    local += __shfl_down(local, off);

  // 3) Cross-wave reduction through LDS: one partial per wave.
  __shared__ float partials[MAX_WAVES];
  const int lane   = tid & (ws - 1);
  const int wave   = tid / ws;
  const int nwaves = (BLOCK + ws - 1) / ws;
  if (lane == 0) partials[wave] = local;
  __syncthreads();

  __shared__ float inv_rms_s;
  if (wave == 0) {
    float total = (lane < nwaves) ? partials[lane] : 0.f;
    for (int off = ws >> 1; off > 0; off >>= 1)
      total += __shfl_down(total, off);
    if (lane == 0) inv_rms_s = rsqrtf(total / H + eps);
  }
  __syncthreads();
  const float inv_rms = inv_rms_s;

  // 4) Rescale (+ optional gamma).
  for (int i = tid; i < H; i += BLOCK) {
    float v = static_cast<float>(xr[i]) * inv_rms;
    if (gamma) v *= static_cast<float>(gamma[i]);
    yr[i] = static_cast<IO>(v);
  }
}

// ---- CPU reference (FP32 math regardless of IO dtype) ----
template <typename IO>
static void rmsnorm_cpu(std::vector<float>& y, const std::vector<IO>& x,
                        const std::vector<IO>& gamma, bool use_gamma, float eps,
                        int R, int H) {
  for (int r = 0; r < R; ++r) {
    double ss = 0.0;
    for (int j = 0; j < H; ++j) {
      float v = static_cast<float>(x[(size_t)r * H + j]);
      ss += (double)v * v;
    }
    float inv = 1.f / std::sqrt((float)(ss / H) + eps);
    for (int j = 0; j < H; ++j) {
      float v = static_cast<float>(x[(size_t)r * H + j]) * inv;
      if (use_gamma) v *= static_cast<float>(gamma[j]);
      y[(size_t)r * H + j] = v;
    }
  }
}

template <typename IO, int BLOCK>
static float run_case(const char* name, int R, int H, bool use_gamma,
                      float eps, bool time_it) {
  std::mt19937 rng(1234 + H);
  std::uniform_real_distribution<float> dist(-2.f, 2.f);

  std::vector<IO> hx((size_t)R * H), hg(H);
  for (auto& v : hx) v = static_cast<IO>(dist(rng));
  for (auto& v : hg) v = static_cast<IO>(0.5f + dist(rng) * 0.1f);

  std::vector<float> ref((size_t)R * H);
  rmsnorm_cpu<IO>(ref, hx, hg, use_gamma, eps, R, H);

  IO *dx, *dy, *dg = nullptr;
  HIP_CHECK(hipMalloc(&dx, hx.size() * sizeof(IO)));
  HIP_CHECK(hipMalloc(&dy, hx.size() * sizeof(IO)));
  HIP_CHECK(hipMemcpy(dx, hx.data(), hx.size() * sizeof(IO),
                      hipMemcpyHostToDevice));
  if (use_gamma) {
    HIP_CHECK(hipMalloc(&dg, hg.size() * sizeof(IO)));
    HIP_CHECK(hipMemcpy(dg, hg.data(), hg.size() * sizeof(IO),
                        hipMemcpyHostToDevice));
  }

  rmsnorm_kernel<IO, BLOCK><<<R, BLOCK>>>(dy, dx, dg, eps, H);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());

  std::vector<IO> hy(hx.size());
  HIP_CHECK(hipMemcpy(hy.data(), dy, hy.size() * sizeof(IO),
                      hipMemcpyDeviceToHost));

  double max_abs = 0.0;
  for (size_t i = 0; i < hy.size(); ++i)
    max_abs = std::max(max_abs, (double)std::fabs((float)hy[i] - ref[i]));

  // fp16 storage has ~2^-11 relative resolution; fp32 path is near-exact.
  float tol = (sizeof(IO) == 2) ? 3e-3f : 1e-4f;
  bool ok = max_abs < tol;

  float ms = 0.f;
  if (time_it) {
    hipEvent_t a, b;
    HIP_CHECK(hipEventCreate(&a));
    HIP_CHECK(hipEventCreate(&b));
    for (int w = 0; w < 5; ++w)
      rmsnorm_kernel<IO, BLOCK><<<R, BLOCK>>>(dy, dx, dg, eps, H);
    HIP_CHECK(hipDeviceSynchronize());
    const int iters = 50;
    HIP_CHECK(hipEventRecord(a));
    for (int it = 0; it < iters; ++it)
      rmsnorm_kernel<IO, BLOCK><<<R, BLOCK>>>(dy, dx, dg, eps, H);
    HIP_CHECK(hipEventRecord(b));
    HIP_CHECK(hipEventSynchronize(b));
    HIP_CHECK(hipEventElapsedTime(&ms, a, b));
    ms /= iters;
    double bytes = 2.0 * (double)R * H * sizeof(IO);  // 1 read + 1 write
    double gbs = bytes / (ms * 1e-3) / 1e9;
    printf("%-22s [%5d x %5d] gamma=%d  max|err|=%.3e  %-4s  %.4f ms  %.0f GB/s\n",
           name, R, H, (int)use_gamma, max_abs, ok ? "PASS" : "FAIL", ms, gbs);
  } else {
    printf("%-22s [%5d x %5d] gamma=%d  max|err|=%.3e  %-4s\n", name, R, H,
           (int)use_gamma, max_abs, ok ? "PASS" : "FAIL");
  }

  HIP_CHECK(hipFree(dx));
  HIP_CHECK(hipFree(dy));
  if (dg) HIP_CHECK(hipFree(dg));
  return ok ? 0.f : 1.f;
}

int main() {
  int dev = 0;
  hipDeviceProp_t prop;
  HIP_CHECK(hipGetDeviceProperties(&prop, dev));
  printf("Device: %s  warpSize=%d\n", prop.name, prop.warpSize);
  printf("---------------------------------------------------------------\n");

  float fails = 0.f;
  // fp32 path
  fails += run_case<float, 256>("rmsnorm fp32", 1024, 4096, true, 1e-6f, true);
  fails += run_case<float, 256>("rmsnorm fp32 no-gamma", 512, 8192, false, 1e-6f, true);
  fails += run_case<float, 128>("rmsnorm fp32 odd-H", 300, 4097, true, 1e-6f, false);
  // fp16 IO path (fp32 accumulate)
  fails += run_case<__half, 256>("rmsnorm fp16 IO", 1024, 4096, true, 1e-6f, true);
  fails += run_case<__half, 512>("rmsnorm fp16 IO big", 256, 16384, true, 1e-6f, true);

  printf("---------------------------------------------------------------\n");
  printf("%s\n", fails == 0.f ? "ALL TESTS PASSED" : "SOME TESTS FAILED");
  return fails == 0.f ? 0 : 1;
}
