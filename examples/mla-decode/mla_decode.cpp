// Portable HIP reference for DeepSeek-style MLA (Multi-Latent Attention) decode.
//
// One decode step (q_len = 1). After "weight absorption" the up-projection of
// the latent KV is folded into the Q and output projections, so all heads attend
// directly against a single shared low-rank latent stream -- structurally this is
// MQA with key width (D_C + D_PE) and value width D_C. See ../../wiki/kernels/mla-decode.md
//
// Per token in the KV history we cache only:
//   c_kv : D_C  latent  (shared across all heads)
//   k_pe : D_PE rope key (shared across all heads)
// The score for head h at position n is:
//   s = sm_scale * ( q_nope[h] . c_kv[n]  +  q_pe[h] . k_pe[n] )
// then online softmax over n, and the output accumulates the latent c_kv:
//   out[h] = sum_n softmax(s)[n] * c_kv[n]      (dim D_C)
//
// This is the exact math AITER's absorbed MLA decode runs; dims are kept tiny but
// realistic in *shape* (latent >> rope, value = latent).  All fp32.
//
// PORTABLE: pure HIP (FMA + LDS + wave shuffle). Builds AND runs on gfx1201.

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>

#define HIP_CHECK(x)                                                            \
  do {                                                                          \
    hipError_t e = (x);                                                         \
    if (e != hipSuccess) {                                                      \
      printf("HIP error %s at %s:%d\n", hipGetErrorString(e), __FILE__,         \
             __LINE__);                                                         \
      exit(1);                                                                  \
    }                                                                           \
  } while (0)

// Realistic *shape* (latent >> rope, value == latent), tiny magnitudes.
constexpr int H    = 16;   // query heads        (DeepSeek-V3: 128)
constexpr int D_C  = 64;   // latent / value dim  (DeepSeek-V3: 512)
constexpr int D_PE = 16;   // decoupled rope dim  (DeepSeek-V3: 64)
constexpr int N    = 256;  // KV history length (tokens already in cache)

constexpr int BLK = 64;    // threads per block (one block per head)

// One block handles one head. Threads cooperatively stream the N latent tokens,
// keeping running max / denom / accumulator in registers + LDS, online-softmax.
__global__ void mla_decode_kernel(const float* __restrict__ q_nope, // [H, D_C]
                                  const float* __restrict__ q_pe,   // [H, D_PE]
                                  const float* __restrict__ kv_c,   // [N, D_C]
                                  const float* __restrict__ k_pe,   // [N, D_PE]
                                  float* __restrict__ out,          // [H, D_C]
                                  float sm_scale) {
  const int h   = blockIdx.x;          // head index
  const int lane = threadIdx.x;        // 0..BLK-1

  // Stage this head's query in LDS so every thread can read all of it.
  __shared__ float sq_nope[D_C];
  __shared__ float sq_pe[D_PE];
  for (int i = lane; i < D_C;  i += BLK) sq_nope[i] = q_nope[h * D_C  + i];
  for (int i = lane; i < D_PE; i += BLK) sq_pe[i]   = q_pe[h * D_PE + i];
  __syncthreads();

  // Online softmax running state (replicated per thread, kept consistent).
  float m_i = -INFINITY;   // running max of scores
  float l_i = 0.0f;        // running softmax denominator
  // Output accumulator over the latent dim, partitioned across threads.
  // Each thread owns latent indices  d = lane, lane+BLK, ...
  float acc[(D_C + BLK - 1) / BLK];
  for (int t = 0; t < (D_C + BLK - 1) / BLK; ++t) acc[t] = 0.0f;

  __shared__ float s_reduce[BLK];

  for (int n = 0; n < N; ++n) {
    const float* kvn = kv_c  + n * D_C;
    const float* kpn = k_pe  + n * D_PE;

    // Partial dot products: q_nope . c_kv  + q_pe . k_pe
    float part = 0.0f;
    for (int i = lane; i < D_C;  i += BLK) part = fmaf(sq_nope[i], kvn[i], part);
    for (int i = lane; i < D_PE; i += BLK) part = fmaf(sq_pe[i],   kpn[i], part);

    // Block reduction of `part` -> full score s for this (h, n).
    s_reduce[lane] = part;
    __syncthreads();
    for (int stride = BLK / 2; stride > 0; stride >>= 1) {
      if (lane < stride) s_reduce[lane] += s_reduce[lane + stride];
      __syncthreads();
    }
    float s = s_reduce[0] * sm_scale;
    __syncthreads();

    // Online-softmax update (same on every thread).
    float m_new = fmaxf(m_i, s);
    float alpha = __expf(m_i - m_new);   // rescale factor for old state
    float p     = __expf(s   - m_new);   // weight of this token
    l_i = l_i * alpha + p;

    // Rescale + accumulate the latent value (value == latent c_kv).
    for (int t = 0, d = lane; d < D_C; ++t, d += BLK)
      acc[t] = acc[t] * alpha + p * kvn[d];

    m_i = m_new;
  }

  // Normalize and write out.
  for (int t = 0, d = lane; d < D_C; ++t, d += BLK)
    out[h * D_C + d] = acc[t] / l_i;
}

// ---- CPU reference ---------------------------------------------------------
static void mla_decode_cpu(const std::vector<float>& q_nope,
                           const std::vector<float>& q_pe,
                           const std::vector<float>& kv_c,
                           const std::vector<float>& k_pe,
                           std::vector<float>& out, float sm_scale) {
  for (int h = 0; h < H; ++h) {
    std::vector<float> s(N);
    float m = -INFINITY;
    for (int n = 0; n < N; ++n) {
      double dot = 0.0;
      for (int i = 0; i < D_C;  ++i) dot += (double)q_nope[h*D_C+i] * kv_c[n*D_C+i];
      for (int i = 0; i < D_PE; ++i) dot += (double)q_pe[h*D_PE+i] * k_pe[n*D_PE+i];
      s[n] = (float)(dot * sm_scale);
      m = fmaxf(m, s[n]);
    }
    double denom = 0.0;
    for (int n = 0; n < N; ++n) { s[n] = expf(s[n] - m); denom += s[n]; }
    for (int d = 0; d < D_C; ++d) {
      double a = 0.0;
      for (int n = 0; n < N; ++n) a += (double)s[n] * kv_c[n*D_C+d];
      out[h*D_C+d] = (float)(a / denom);
    }
  }
}

int main() {
  printf("MLA decode (absorbed, low-rank latent KV) -- portable HIP, fp32\n");
  printf("  H=%d heads, D_C=%d latent, D_PE=%d rope, N=%d KV tokens\n",
         H, D_C, D_PE, N);

  std::mt19937 rng(1234);
  std::uniform_real_distribution<float> dist(-0.5f, 0.5f);

  std::vector<float> q_nope(H*D_C), q_pe(H*D_PE), kv_c(N*D_C), k_pe(N*D_PE);
  for (auto& v : q_nope) v = dist(rng);
  for (auto& v : q_pe)   v = dist(rng);
  for (auto& v : kv_c)   v = dist(rng);
  for (auto& v : k_pe)   v = dist(rng);

  const float sm_scale = 1.0f / std::sqrt((float)(D_C + D_PE));

  std::vector<float> out_ref(H*D_C), out_gpu(H*D_C);
  mla_decode_cpu(q_nope, q_pe, kv_c, k_pe, out_ref, sm_scale);

  float *dqn, *dqp, *dkv, *dkp, *dout;
  HIP_CHECK(hipMalloc(&dqn,  q_nope.size()*sizeof(float)));
  HIP_CHECK(hipMalloc(&dqp,  q_pe.size()  *sizeof(float)));
  HIP_CHECK(hipMalloc(&dkv,  kv_c.size()  *sizeof(float)));
  HIP_CHECK(hipMalloc(&dkp,  k_pe.size()  *sizeof(float)));
  HIP_CHECK(hipMalloc(&dout, out_gpu.size()*sizeof(float)));
  HIP_CHECK(hipMemcpy(dqn, q_nope.data(), q_nope.size()*sizeof(float), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(dqp, q_pe.data(),   q_pe.size()  *sizeof(float), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(dkv, kv_c.data(),   kv_c.size()  *sizeof(float), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(dkp, k_pe.data(),   k_pe.size()  *sizeof(float), hipMemcpyHostToDevice));

  dim3 grid(H), block(BLK);

  // Warmup + correctness run.
  mla_decode_kernel<<<grid, block>>>(dqn, dqp, dkv, dkp, dout, sm_scale);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());
  HIP_CHECK(hipMemcpy(out_gpu.data(), dout, out_gpu.size()*sizeof(float), hipMemcpyDeviceToHost));

  // Timing.
  const int iters = 1000;
  hipEvent_t t0, t1;
  HIP_CHECK(hipEventCreate(&t0));
  HIP_CHECK(hipEventCreate(&t1));
  HIP_CHECK(hipEventRecord(t0));
  for (int it = 0; it < iters; ++it)
    mla_decode_kernel<<<grid, block>>>(dqn, dqp, dkv, dkp, dout, sm_scale);
  HIP_CHECK(hipEventRecord(t1));
  HIP_CHECK(hipEventSynchronize(t1));
  float ms = 0.0f;
  HIP_CHECK(hipEventElapsedTime(&ms, t0, t1));
  float us = ms * 1000.0f / iters;

  // Self-check.
  float max_abs = 0.0f, max_rel = 0.0f;
  for (size_t i = 0; i < out_ref.size(); ++i) {
    float a = std::fabs(out_gpu[i] - out_ref[i]);
    max_abs = fmaxf(max_abs, a);
    max_rel = fmaxf(max_rel, a / (std::fabs(out_ref[i]) + 1e-6f));
  }

  // Bytes streamed per decode = the KV history (latent + rope), the memory-bound term.
  double bytes = (double)N * (D_C + D_PE) * sizeof(float);
  double gbps  = bytes / (us * 1e-6) / 1e9;

  printf("  per-decode: %.2f us   KV-stream BW: %.1f GB/s\n", us, gbps);
  printf("  max_abs_err = %.3e   max_rel_err = %.3e\n", max_abs, max_rel);

  bool ok = max_abs < 1e-4f;
  printf("%s\n", ok ? "PASS" : "FAIL");

  HIP_CHECK(hipFree(dqn)); HIP_CHECK(hipFree(dqp)); HIP_CHECK(hipFree(dkv));
  HIP_CHECK(hipFree(dkp)); HIP_CHECK(hipFree(dout));
  return ok ? 0 : 1;
}
