// preshuffle_gemm_rocwmma.cpp
//
// PORTABLE demonstration of the "preshuffle the weights" idea from the FlyDSL
// 04-preshuffle_gemm.py tutorial. The source uses rocWMMA's fragment API; on
// gfx950 the compiler emits MFMA instructions. Both kernels run and self-check.
//
// Idea being demonstrated
// -----------------------
// A wavefront matrix instruction does not consume a plain row-major tile: each
// lane wants a specific 16x16 fragment of B. Two ways to feed it:
//   1) load row-major B with a big leading dim (ldb = N) per output tile -- the
//      hardware fragment loader strides across the whole matrix every k-step.
//   2) PRESHUFFLE B once on the host into fragment order: store every 16x16
//      (k,n) tile contiguously. The steady-state kernel then reads each fragment
//      from a flat 16x16 block (ldb = 16) -- a contiguous, coalesced copy with
//      no per-tile address math.
//
// This file builds BOTH kernels, runs them on the GPU, and checks each against
// a CPU reference. The preshuffled kernel is the one that mirrors FlyDSL's
// `copy(Bsh.tile(...), sB.next())` flat-copy path.

#include <hip/hip_runtime.h>
#include <rocwmma/rocwmma.hpp>

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>

using namespace rocwmma;

#define HIP_CHECK(x)                                                            \
  do {                                                                          \
    hipError_t e = (x);                                                         \
    if (e != hipSuccess) {                                                      \
      printf("HIP error %s at line %d: %s\n", #x, __LINE__,                     \
             hipGetErrorString(e));                                            \
      std::exit(1);                                                             \
    }                                                                           \
  } while (0)

constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

// ---------------------------------------------------------------------------
// Kernel 1: naive row-major B. Each wave computes one 16x16 output tile and
// loads B fragments straight out of the KxN row-major matrix (ldb = N).
// ---------------------------------------------------------------------------
__global__ void gemm_rowmajor(const __half* __restrict__ A,   // M x K, row-major
                              const __half* __restrict__ B,   // K x N, row-major
                              float* __restrict__ C,          // M x N, row-major
                              int M, int N, int K) {
  int tileM = blockIdx.y;   // which 16-row block
  int tileN = blockIdx.x;   // which 16-col block

  auto fragC = fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float>();
  fill_fragment(fragC, 0.0f);

  for (int k0 = 0; k0 < K; k0 += WMMA_K) {
    auto fragA = fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, row_major>();
    auto fragB = fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, row_major>();

    const __half* aptr = A + (tileM * WMMA_M) * K + k0;          // ld = K
    const __half* bptr = B + (k0) * N + (tileN * WMMA_N);        // ld = N  <-- strided
    load_matrix_sync(fragA, aptr, K);
    load_matrix_sync(fragB, bptr, N);
    mma_sync(fragC, fragA, fragB, fragC);
  }

  float* cptr = C + (tileM * WMMA_M) * N + (tileN * WMMA_N);
  store_matrix_sync(cptr, fragC, N, mem_row_major);
}

// ---------------------------------------------------------------------------
// Kernel 2: PRESHUFFLED B. B has been packed on the host so that every 16x16
// (k,n) tile is stored contiguously, row-major inside the tile, with tiles
// ordered (tileK, tileN). For output tile (tileM,tileN) the k-loop reads tile
// (k0/16, tileN) as a flat 16x16 block: ldb = 16, address = base + offset.
// This is the FlyDSL `copy(Bsh.tile(...))` flat path -- no strided gather.
// ---------------------------------------------------------------------------
__global__ void gemm_preshuffled(const __half* __restrict__ A,    // M x K, row-major
                                 const __half* __restrict__ Bsh,  // packed fragments
                                 float* __restrict__ C,           // M x N, row-major
                                 int M, int N, int K) {
  int tileM = blockIdx.y;
  int tileN = blockIdx.x;
  int nTilesK = K / WMMA_K;

  auto fragC = fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float>();
  fill_fragment(fragC, 0.0f);

  for (int kt = 0; kt < nTilesK; ++kt) {
    auto fragA = fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, row_major>();
    auto fragB = fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, row_major>();

    const __half* aptr = A + (tileM * WMMA_M) * K + (kt * WMMA_K);  // ld = K
    // Flat 16x16 fragment: contiguous block, leading dim = 16.
    const __half* bptr = Bsh + ((size_t)(kt * (N / WMMA_N) + tileN)) * (WMMA_K * WMMA_N);
    load_matrix_sync(fragA, aptr, K);
    load_matrix_sync(fragB, bptr, WMMA_N);   // ld = 16  <-- contiguous fragment
    mma_sync(fragC, fragA, fragB, fragC);
  }

  float* cptr = C + (tileM * WMMA_M) * N + (tileN * WMMA_N);
  store_matrix_sync(cptr, fragC, N, mem_row_major);
}

// Host-side preshuffle: KxN row-major B -> tile-contiguous fragment order.
// Output index: ((tileK * nTilesN + tileN) * 256) + (kk * 16 + nn)
static void preshuffle_B(const std::vector<__half>& B, std::vector<__half>& Bsh,
                         int K, int N) {
  int nTilesK = K / WMMA_K;
  int nTilesN = N / WMMA_N;
  Bsh.resize((size_t)K * N);
  for (int tk = 0; tk < nTilesK; ++tk)
    for (int tn = 0; tn < nTilesN; ++tn) {
      size_t base = ((size_t)(tk * nTilesN + tn)) * (WMMA_K * WMMA_N);
      for (int kk = 0; kk < WMMA_K; ++kk)
        for (int nn = 0; nn < WMMA_N; ++nn) {
          int gk = tk * WMMA_K + kk;
          int gn = tn * WMMA_N + nn;
          Bsh[base + kk * WMMA_N + nn] = B[(size_t)gk * N + gn];
        }
    }
}

int main() {
  const int M = 256, N = 256, K = 256;
  printf("Preshuffle GEMM demo (rocWMMA 16x16x16, fp16->fp32)  M=N=K=%d\n", M);

  std::mt19937 rng(123);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  std::vector<__half> hA((size_t)M * K), hB((size_t)K * N);
  for (auto& x : hA) x = __float2half(dist(rng));
  for (auto& x : hB) x = __float2half(dist(rng));

  std::vector<__half> hBsh;
  preshuffle_B(hB, hBsh, K, N);

  // CPU reference (fp32 accumulate).
  std::vector<float> ref((size_t)M * N, 0.0f);
  for (int m = 0; m < M; ++m)
    for (int k = 0; k < K; ++k) {
      float a = __half2float(hA[(size_t)m * K + k]);
      for (int n = 0; n < N; ++n)
        ref[(size_t)m * N + n] += a * __half2float(hB[(size_t)k * N + n]);
    }

  __half *dA, *dB, *dBsh;
  float *dC;
  HIP_CHECK(hipMalloc(&dA, hA.size() * sizeof(__half)));
  HIP_CHECK(hipMalloc(&dB, hB.size() * sizeof(__half)));
  HIP_CHECK(hipMalloc(&dBsh, hBsh.size() * sizeof(__half)));
  HIP_CHECK(hipMalloc(&dC, (size_t)M * N * sizeof(float)));
  HIP_CHECK(hipMemcpy(dA, hA.data(), hA.size() * sizeof(__half), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(dB, hB.data(), hB.size() * sizeof(__half), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(dBsh, hBsh.data(), hBsh.size() * sizeof(__half), hipMemcpyHostToDevice));

  // One wave per 16x16 output tile. Query the wave size at runtime because
  // warpSize is device-only and is not valid in host code.
  int dev = 0;
  hipDeviceProp_t prop;
  HIP_CHECK(hipGetDeviceProperties(&prop, dev));
  dim3 block(prop.warpSize, 1, 1);
  dim3 grid(N / WMMA_N, M / WMMA_M, 1);

  auto check = [&](const char* name, std::vector<float>& hC) -> bool {
    float maxErr = 0.0f;
    for (size_t i = 0; i < hC.size(); ++i)
      maxErr = fmaxf(maxErr, fabsf(hC[i] - ref[i]));
    bool ok = maxErr < 0.5f;  // fp16 inputs, K=256 -> generous tolerance
    printf("  %-12s max abs err = %.4f  ->  %s\n", name, maxErr,
           ok ? "PASS" : "FAIL");
    return ok;
  };

  auto time_kernel = [&](void (*kern)(const __half*, const __half*, float*, int,
                                      int, int),
                         const __half* Barg, std::vector<float>& hC) -> float {
    HIP_CHECK(hipMemset(dC, 0, (size_t)M * N * sizeof(float)));
    // warmup
    hipLaunchKernelGGL(kern, grid, block, 0, 0, dA, Barg, dC, M, N, K);
    HIP_CHECK(hipDeviceSynchronize());
    hipEvent_t evStart, evStop;
    HIP_CHECK(hipEventCreate(&evStart));
    HIP_CHECK(hipEventCreate(&evStop));
    const int iters = 200;
    HIP_CHECK(hipEventRecord(evStart));
    for (int i = 0; i < iters; ++i)
      hipLaunchKernelGGL(kern, grid, block, 0, 0, dA, Barg, dC, M, N, K);
    HIP_CHECK(hipEventRecord(evStop));
    HIP_CHECK(hipEventSynchronize(evStop));
    float ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&ms, evStart, evStop));
    hC.resize((size_t)M * N);
    HIP_CHECK(hipMemcpy(hC.data(), dC, hC.size() * sizeof(float), hipMemcpyDeviceToHost));
    HIP_CHECK(hipEventDestroy(evStart));
    HIP_CHECK(hipEventDestroy(evStop));
    return ms / iters;
  };

  std::vector<float> hC1, hC2;
  float t1 = time_kernel(gemm_rowmajor, dB, hC1);
  float t2 = time_kernel(gemm_preshuffled, dBsh, hC2);

  printf("Correctness:\n");
  bool ok1 = check("row-major", hC1);
  bool ok2 = check("preshuffled", hC2);

  double flops = 2.0 * M * N * K;
  printf("Timing (avg over 200 iters):\n");
  printf("  row-major    %.4f ms  (%.1f GFLOP/s)\n", t1, flops / (t1 * 1e6));
  printf("  preshuffled  %.4f ms  (%.1f GFLOP/s)\n", t2, flops / (t2 * 1e6));

  HIP_CHECK(hipFree(dA));
  HIP_CHECK(hipFree(dB));
  HIP_CHECK(hipFree(dBsh));
  HIP_CHECK(hipFree(dC));

  bool ok = ok1 && ok2;
  printf("\nRESULT: %s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
