// Portable HIP fused-MoE reference (fp32), verified on gfx950.
//
// One thread-block processes one token end to end:
//   1. router GEMV  : logits[e] = X . Wrouter[e]   (e in 0..E)
//   2. top-k gating : pick top_k experts, softmax over the selected logits
//   3. for each selected expert (fused, no HBM round-trip for h):
//        gate[n] = X . Wgate[e][n]                 (gate-up GEMV)
//        up[n]   = X . Wup[e][n]
//        h[n]    = SiLU(gate[n]) * up[n]           (activation, kept in LDS)
//        y[k]   += g_e * (h . Wdown[e][k])         (down GEMV, weighted reduce)
//   4. write y[k] to out[token]
//
// This mirrors the structure on the wiki page (gate-up + SiLU + down) but in
// pure HIP fp32 so it is portable and self-checking. Production MoE replaces the
// per-token GEMVs with grouped GEMM on MFMA/WMMA matrix cores and FP8 weights.

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include <random>

#define HIP_CHECK(x)                                                            \
  do {                                                                          \
    hipError_t e_ = (x);                                                        \
    if (e_ != hipSuccess) {                                                     \
      printf("HIP error %s at %s:%d\n", hipGetErrorString(e_), __FILE__,        \
             __LINE__);                                                         \
      exit(1);                                                                  \
    }                                                                           \
  } while (0)

// Problem dimensions (small so it runs instantly and verifies easily).
constexpr int T     = 64;   // tokens
constexpr int D     = 128;  // model dim (GEMV K)
constexpr int N     = 256;  // expert hidden dim
constexpr int E     = 8;    // number of experts
constexpr int TOPK  = 2;    // experts selected per token
constexpr int BLOCK = 256;  // threads per block (>= N for the LDS h buffer)

__device__ __forceinline__ float silu(float v) {
  return v / (1.0f + expf(-v));
}

// One block == one token.
__global__ void fused_moe_kernel(const float* __restrict__ X,        // [T, D]
                                 const float* __restrict__ Wrouter,  // [E, D]
                                 const float* __restrict__ Wgate,    // [E, N, D]
                                 const float* __restrict__ Wup,      // [E, N, D]
                                 const float* __restrict__ Wdown,    // [E, D, N]
                                 float* __restrict__ Out,            // [T, D]
                                 int Tn, int Dn, int Nn, int En, int topk) {
  const int tok = blockIdx.x;
  const int tid = threadIdx.x;
  if (tok >= Tn) return;

  __shared__ float s_x[D];          // this token's activation
  __shared__ float s_logits[E];     // router logits
  __shared__ float s_h[N];          // SiLU(gate)*up for the current expert
  __shared__ int   s_topk_e[TOPK];  // selected expert ids
  __shared__ float s_topk_w[TOPK];  // softmax gate weights
  __shared__ float s_y[D];          // accumulated output for this token

  // Load activation row into LDS.
  for (int i = tid; i < Dn; i += blockDim.x) s_x[i] = X[tok * Dn + i];
  for (int i = tid; i < Dn; i += blockDim.x) s_y[i] = 0.0f;
  __syncthreads();

  // --- 1. router GEMV: one thread per expert (E is small). ---
  if (tid < En) {
    float acc = 0.0f;
    const float* w = Wrouter + tid * Dn;
    for (int k = 0; k < Dn; ++k) acc += s_x[k] * w[k];
    s_logits[tid] = acc;
  }
  __syncthreads();

  // --- 2. top-k selection + softmax over selected logits (single thread). ---
  if (tid == 0) {
    bool used[E];
    for (int e = 0; e < En; ++e) used[e] = false;
    for (int s = 0; s < topk; ++s) {
      int best = -1;
      float bestv = -INFINITY;
      for (int e = 0; e < En; ++e) {
        if (!used[e] && s_logits[e] > bestv) { bestv = s_logits[e]; best = e; }
      }
      used[best] = true;
      s_topk_e[s] = best;
    }
    // softmax over the selected logits only (standard MoE router-prob renorm).
    float mx = -INFINITY;
    for (int s = 0; s < topk; ++s) mx = fmaxf(mx, s_logits[s_topk_e[s]]);
    float sum = 0.0f;
    for (int s = 0; s < topk; ++s) sum += expf(s_logits[s_topk_e[s]] - mx);
    for (int s = 0; s < topk; ++s)
      s_topk_w[s] = expf(s_logits[s_topk_e[s]] - mx) / sum;
  }
  __syncthreads();

  // --- 3. per-expert fused gate-up + SiLU + down. ---
  for (int s = 0; s < topk; ++s) {
    const int e   = s_topk_e[s];
    const float g = s_topk_w[s];
    const float* wg = Wgate + (size_t)e * Nn * Dn;
    const float* wu = Wup   + (size_t)e * Nn * Dn;

    // gate-up GEMV + SiLU*mul -> s_h[n], one thread per hidden unit.
    for (int n = tid; n < Nn; n += blockDim.x) {
      float ag = 0.0f, au = 0.0f;
      const float* rg = wg + (size_t)n * Dn;
      const float* ru = wu + (size_t)n * Dn;
      for (int k = 0; k < Dn; ++k) {
        float xk = s_x[k];
        ag += xk * rg[k];
        au += xk * ru[k];
      }
      s_h[n] = silu(ag) * au;
    }
    __syncthreads();

    // down GEMV: y[d] += g * (h . Wdown[e][d]); one thread per output dim.
    const float* wd = Wdown + (size_t)e * Dn * Nn;  // [D, N]
    for (int d = tid; d < Dn; d += blockDim.x) {
      float acc = 0.0f;
      const float* rd = wd + (size_t)d * Nn;
      for (int n = 0; n < Nn; ++n) acc += s_h[n] * rd[n];
      s_y[d] += g * acc;
    }
    __syncthreads();
  }

  // --- 4. write output. ---
  for (int d = tid; d < Dn; d += blockDim.x) Out[tok * Dn + d] = s_y[d];
}

// ----------------------------- CPU reference -----------------------------
static void cpu_reference(const std::vector<float>& X,
                          const std::vector<float>& Wrouter,
                          const std::vector<float>& Wgate,
                          const std::vector<float>& Wup,
                          const std::vector<float>& Wdown,
                          std::vector<float>& Out) {
  for (int t = 0; t < T; ++t) {
    // router
    std::vector<float> logits(E, 0.0f);
    for (int e = 0; e < E; ++e) {
      float acc = 0.0f;
      for (int k = 0; k < D; ++k) acc += X[t * D + k] * Wrouter[e * D + k];
      logits[e] = acc;
    }
    // top-k
    std::vector<int> idx(E);
    for (int e = 0; e < E; ++e) idx[e] = e;
    std::partial_sort(idx.begin(), idx.begin() + TOPK, idx.end(),
                      [&](int a, int b) { return logits[a] > logits[b]; });
    std::vector<int> sel(idx.begin(), idx.begin() + TOPK);
    // softmax over selected
    float mx = -INFINITY;
    for (int s = 0; s < TOPK; ++s) mx = std::max(mx, logits[sel[s]]);
    float sum = 0.0f;
    for (int s = 0; s < TOPK; ++s) sum += std::exp(logits[sel[s]] - mx);
    std::vector<float> w(TOPK);
    for (int s = 0; s < TOPK; ++s) w[s] = std::exp(logits[sel[s]] - mx) / sum;

    std::vector<float> y(D, 0.0f);
    for (int s = 0; s < TOPK; ++s) {
      int e = sel[s];
      std::vector<float> h(N);
      for (int n = 0; n < N; ++n) {
        float ag = 0.0f, au = 0.0f;
        for (int k = 0; k < D; ++k) {
          float xk = X[t * D + k];
          ag += xk * Wgate[((size_t)e * N + n) * D + k];
          au += xk * Wup[((size_t)e * N + n) * D + k];
        }
        float sg = ag / (1.0f + std::exp(-ag));
        h[n] = sg * au;
      }
      for (int d = 0; d < D; ++d) {
        float acc = 0.0f;
        for (int n = 0; n < N; ++n) acc += h[n] * Wdown[((size_t)e * D + d) * N + n];
        y[d] += w[s] * acc;
      }
    }
    for (int d = 0; d < D; ++d) Out[t * D + d] = y[d];
  }
}

int main() {
  std::mt19937 rng(1234);
  std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

  std::vector<float> X((size_t)T * D);
  std::vector<float> Wrouter((size_t)E * D);
  std::vector<float> Wgate((size_t)E * N * D);
  std::vector<float> Wup((size_t)E * N * D);
  std::vector<float> Wdown((size_t)E * D * N);
  for (auto& v : X) v = dist(rng);
  for (auto& v : Wrouter) v = dist(rng);
  for (auto& v : Wgate) v = dist(rng) * 0.25f;
  for (auto& v : Wup)   v = dist(rng) * 0.25f;
  for (auto& v : Wdown) v = dist(rng) * 0.25f;

  std::vector<float> Out_gpu((size_t)T * D, 0.0f);
  std::vector<float> Out_cpu((size_t)T * D, 0.0f);

  float *dX, *dWr, *dWg, *dWu, *dWd, *dOut;
  HIP_CHECK(hipMalloc(&dX,  X.size()       * sizeof(float)));
  HIP_CHECK(hipMalloc(&dWr, Wrouter.size() * sizeof(float)));
  HIP_CHECK(hipMalloc(&dWg, Wgate.size()   * sizeof(float)));
  HIP_CHECK(hipMalloc(&dWu, Wup.size()     * sizeof(float)));
  HIP_CHECK(hipMalloc(&dWd, Wdown.size()   * sizeof(float)));
  HIP_CHECK(hipMalloc(&dOut, Out_gpu.size() * sizeof(float)));

  HIP_CHECK(hipMemcpy(dX,  X.data(),       X.size()*sizeof(float),       hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(dWr, Wrouter.data(), Wrouter.size()*sizeof(float), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(dWg, Wgate.data(),   Wgate.size()*sizeof(float),   hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(dWu, Wup.data(),     Wup.size()*sizeof(float),     hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(dWd, Wdown.data(),   Wdown.size()*sizeof(float),   hipMemcpyHostToDevice));

  dim3 grid(T), block(BLOCK);

  // warmup + correctness launch
  fused_moe_kernel<<<grid, block>>>(dX, dWr, dWg, dWu, dWd, dOut, T, D, N, E, TOPK);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());

  // timing
  hipEvent_t t0, t1;
  HIP_CHECK(hipEventCreate(&t0));
  HIP_CHECK(hipEventCreate(&t1));
  const int iters = 200;
  HIP_CHECK(hipEventRecord(t0));
  for (int i = 0; i < iters; ++i)
    fused_moe_kernel<<<grid, block>>>(dX, dWr, dWg, dWu, dWd, dOut, T, D, N, E, TOPK);
  HIP_CHECK(hipEventRecord(t1));
  HIP_CHECK(hipEventSynchronize(t1));
  float ms = 0.0f;
  HIP_CHECK(hipEventElapsedTime(&ms, t0, t1));

  HIP_CHECK(hipMemcpy(Out_gpu.data(), dOut, Out_gpu.size()*sizeof(float), hipMemcpyDeviceToHost));

  cpu_reference(X, Wrouter, Wgate, Wup, Wdown, Out_cpu);

  double max_abs = 0.0, max_rel = 0.0;
  for (size_t i = 0; i < Out_cpu.size(); ++i) {
    double a = Out_gpu[i], b = Out_cpu[i];
    double e = std::fabs(a - b);
    max_abs = std::max(max_abs, e);
    max_rel = std::max(max_rel, e / (std::fabs(b) + 1e-6));
  }

  printf("Fused MoE (fp32, portable HIP)\n");
  printf("  dims: T=%d D=%d N=%d E=%d top_k=%d\n", T, D, N, E, TOPK);
  printf("  kernel time: %.3f us/iter (%d iters)\n", ms * 1000.0 / iters, iters);
  printf("  max abs err: %.3e\n", max_abs);
  printf("  max rel err: %.3e\n", max_rel);

  bool pass = max_abs < 1e-3;
  printf("%s\n", pass ? "PASS" : "FAIL");

  HIP_CHECK(hipFree(dX));  HIP_CHECK(hipFree(dWr)); HIP_CHECK(hipFree(dWg));
  HIP_CHECK(hipFree(dWu)); HIP_CHECK(hipFree(dWd)); HIP_CHECK(hipFree(dOut));
  return pass ? 0 : 1;
}
