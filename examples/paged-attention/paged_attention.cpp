// Portable HIP paged-attention DECODE reference (fp32).
//
// Single query step (q_len = 1) attends over a paged KV cache addressed through
// a per-sequence block table (page-table indirection, exactly like vLLM/AITER).
// One workgroup handles one (sequence, query-head) pair; the block table maps
// logical KV blocks to scattered physical blocks in HBM. Online-softmax
// (FlashAttention) recurrence runs across the gathered KV blocks.
//
// This is PORTABLE pure-HIP (FMA math + LDS reduction + warp shuffles): it BUILDS
// AND RUNS on gfx1201 (RDNA4) and is verified against a CPU reference. fp32.
//
// Layout (mirrors the wiki page):
//   q          : [num_seqs, num_q_heads, HEAD_DIM]
//   k_cache    : [num_blocks, num_kv_heads, BLOCK_SIZE, HEAD_DIM]
//   v_cache    : [num_blocks, num_kv_heads, BLOCK_SIZE, HEAD_DIM]
//   block_table: [num_seqs, max_num_blocks]   (logical block -> physical block)
//   seq_lens   : [num_seqs]
//   out        : [num_seqs, num_q_heads, HEAD_DIM]
// GQA: num_q_heads = num_kv_heads * GROUP.

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>
#include <algorithm>

#define HIP_CHECK(c)                                                            \
  do {                                                                          \
    hipError_t e = (c);                                                         \
    if (e != hipSuccess) {                                                      \
      fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(e),         \
              __FILE__, __LINE__);                                              \
      std::exit(1);                                                            \
    }                                                                           \
  } while (0)

// ---- problem dimensions (compile-time for the kernel) ----
constexpr int HEAD_DIM   = 64;
constexpr int BLOCK_SIZE = 16;   // KV tokens per page
constexpr int GROUP      = 4;    // q heads per kv head (GQA)
constexpr int THREADS    = HEAD_DIM; // one thread per head-dim element

// One block = one (sequence, q_head). blockDim = HEAD_DIM threads.
__global__ void paged_attention_decode_kernel(
    const float* __restrict__ q,           // [num_seqs, num_q_heads, HEAD_DIM]
    const float* __restrict__ k_cache,     // [num_blocks, num_kv_heads, BLOCK_SIZE, HEAD_DIM]
    const float* __restrict__ v_cache,     // same shape as k_cache
    const int*   __restrict__ block_table, // [num_seqs, max_num_blocks]
    const int*   __restrict__ seq_lens,    // [num_seqs]
    float*       __restrict__ out,         // [num_seqs, num_q_heads, HEAD_DIM]
    float scale,
    int num_q_heads, int num_kv_heads, int max_num_blocks)
{
  const int seq    = blockIdx.x;
  const int q_head = blockIdx.y;
  const int kv_head = q_head / GROUP;
  const int d = threadIdx.x;            // 0 .. HEAD_DIM-1

  const int seq_len = seq_lens[seq];
  const int num_logical_blocks = (seq_len + BLOCK_SIZE - 1) / BLOCK_SIZE;

  // This thread owns query element q[d] for (seq, q_head).
  const float qd =
      q[(seq * num_q_heads + q_head) * HEAD_DIM + d] * scale;

  // Online-softmax running state (scalar; broadcast across threads via LDS).
  float m_i = -INFINITY;  // running max of logits
  float l_i = 0.0f;       // running sum of exp
  float acc = 0.0f;       // running output accumulator for element d

  __shared__ float s_q[HEAD_DIM];   // scaled query row
  __shared__ float s_logit;         // current logit (dot product result)
  __shared__ float s_reduce[HEAD_DIM];

  s_q[d] = qd;
  __syncthreads();

  for (int lb = 0; lb < num_logical_blocks; ++lb) {
    const int phys = block_table[seq * max_num_blocks + lb];
    const int tokens_in_block =
        min(BLOCK_SIZE, seq_len - lb * BLOCK_SIZE);

    for (int t = 0; t < tokens_in_block; ++t) {
      // base offset of this KV token's HEAD_DIM vector
      const long base =
          (((long)phys * num_kv_heads + kv_head) * BLOCK_SIZE + t) * HEAD_DIM;

      // ---- QK^T dot product: sum_d s_q[d] * k[d] ----
      const float prod = s_q[d] * k_cache[base + d];
      s_reduce[d] = prod;
      __syncthreads();
      // tree reduction over HEAD_DIM threads
      for (int stride = HEAD_DIM / 2; stride > 0; stride >>= 1) {
        if (d < stride) s_reduce[d] += s_reduce[d + stride];
        __syncthreads();
      }
      if (d == 0) s_logit = s_reduce[0];
      __syncthreads();
      const float logit = s_logit;

      // ---- online softmax update ----
      const float m_new = fmaxf(m_i, logit);
      const float alpha = __expf(m_i - m_new);   // rescale old state
      const float p     = __expf(logit - m_new); // weight of this token
      l_i = l_i * alpha + p;
      acc = acc * alpha + p * v_cache[base + d];
      m_i = m_new;
      __syncthreads();
    }
  }

  // normalize and write one output row element
  const float o = (l_i > 0.0f) ? acc / l_i : 0.0f;
  out[(seq * num_q_heads + q_head) * HEAD_DIM + d] = o;
}

// ----------------- CPU reference -----------------
static void cpu_reference(
    const std::vector<float>& q,
    const std::vector<float>& k_cache,
    const std::vector<float>& v_cache,
    const std::vector<int>&   block_table,
    const std::vector<int>&   seq_lens,
    std::vector<float>&       out,
    float scale,
    int num_seqs, int num_q_heads, int num_kv_heads, int max_num_blocks)
{
  for (int seq = 0; seq < num_seqs; ++seq) {
    const int seq_len = seq_lens[seq];
    const int nlb = (seq_len + BLOCK_SIZE - 1) / BLOCK_SIZE;
    for (int qh = 0; qh < num_q_heads; ++qh) {
      const int kvh = qh / GROUP;
      const float* qrow = &q[(seq * num_q_heads + qh) * HEAD_DIM];

      std::vector<float> logits;
      logits.reserve(seq_len);
      for (int lb = 0; lb < nlb; ++lb) {
        const int phys = block_table[seq * max_num_blocks + lb];
        const int tib = std::min(BLOCK_SIZE, seq_len - lb * BLOCK_SIZE);
        for (int t = 0; t < tib; ++t) {
          const long base =
              (((long)phys * num_kv_heads + kvh) * BLOCK_SIZE + t) * HEAD_DIM;
          float dot = 0.0f;
          for (int d = 0; d < HEAD_DIM; ++d)
            dot += qrow[d] * scale * k_cache[base + d];
          logits.push_back(dot);
        }
      }
      // softmax
      float m = -INFINITY;
      for (float x : logits) m = std::max(m, x);
      float l = 0.0f;
      for (float x : logits) l += std::exp(x - m);
      // weighted sum of V
      float* orow = &out[(seq * num_q_heads + qh) * HEAD_DIM];
      for (int d = 0; d < HEAD_DIM; ++d) orow[d] = 0.0f;
      int idx = 0;
      for (int lb = 0; lb < nlb; ++lb) {
        const int phys = block_table[seq * max_num_blocks + lb];
        const int tib = std::min(BLOCK_SIZE, seq_len - lb * BLOCK_SIZE);
        for (int t = 0; t < tib; ++t, ++idx) {
          const long base =
              (((long)phys * num_kv_heads + kvh) * BLOCK_SIZE + t) * HEAD_DIM;
          const float w = std::exp(logits[idx] - m) / l;
          for (int d = 0; d < HEAD_DIM; ++d)
            orow[d] += w * v_cache[base + d];
        }
      }
    }
  }
}

int main() {
  const int num_seqs      = 3;
  const int num_kv_heads  = 2;
  const int num_q_heads   = num_kv_heads * GROUP; // 8
  const int seq_lens_h[3] = {40, 17, 64};         // ragged (partial tail blocks)
  const int max_seq_len   = 64;
  const int max_num_blocks = (max_seq_len + BLOCK_SIZE - 1) / BLOCK_SIZE; // 4
  const float scale = 1.0f / std::sqrt((float)HEAD_DIM);

  // Total physical blocks: allocate generously and scatter the block tables.
  const int num_blocks = 32;

  std::mt19937 rng(1234);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  std::vector<int> seq_lens(seq_lens_h, seq_lens_h + num_seqs);

  std::vector<float> q((size_t)num_seqs * num_q_heads * HEAD_DIM);
  for (auto& x : q) x = dist(rng);

  std::vector<float> k_cache((size_t)num_blocks * num_kv_heads * BLOCK_SIZE * HEAD_DIM);
  std::vector<float> v_cache(k_cache.size());
  for (auto& x : k_cache) x = dist(rng);
  for (auto& x : v_cache) x = dist(rng);

  // Block tables: scatter logical blocks to non-contiguous physical blocks to
  // exercise the page-table indirection (this is the whole point of paged attn).
  std::vector<int> block_table((size_t)num_seqs * max_num_blocks, 0);
  std::vector<int> phys_pool(num_blocks);
  for (int i = 0; i < num_blocks; ++i) phys_pool[i] = i;
  std::shuffle(phys_pool.begin(), phys_pool.end(), rng);
  int next = 0;
  for (int s = 0; s < num_seqs; ++s) {
    int nlb = (seq_lens[s] + BLOCK_SIZE - 1) / BLOCK_SIZE;
    for (int b = 0; b < nlb; ++b)
      block_table[s * max_num_blocks + b] = phys_pool[next++];
  }

  // Device buffers
  float *d_q, *d_k, *d_v, *d_out;
  int *d_bt, *d_sl;
  HIP_CHECK(hipMalloc(&d_q, q.size() * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_k, k_cache.size() * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_v, v_cache.size() * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_out, q.size() * sizeof(float)));
  HIP_CHECK(hipMalloc(&d_bt, block_table.size() * sizeof(int)));
  HIP_CHECK(hipMalloc(&d_sl, seq_lens.size() * sizeof(int)));

  HIP_CHECK(hipMemcpy(d_q, q.data(), q.size() * sizeof(float), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_k, k_cache.data(), k_cache.size() * sizeof(float), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_v, v_cache.data(), v_cache.size() * sizeof(float), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_bt, block_table.data(), block_table.size() * sizeof(int), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_sl, seq_lens.data(), seq_lens.size() * sizeof(int), hipMemcpyHostToDevice));

  dim3 grid(num_seqs, num_q_heads);
  dim3 block(THREADS);

  // Warmup + timed launch
  hipEvent_t t0, t1;
  HIP_CHECK(hipEventCreate(&t0));
  HIP_CHECK(hipEventCreate(&t1));

  paged_attention_decode_kernel<<<grid, block>>>(
      d_q, d_k, d_v, d_bt, d_sl, d_out, scale,
      num_q_heads, num_kv_heads, max_num_blocks);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());

  const int iters = 200;
  HIP_CHECK(hipEventRecord(t0));
  for (int i = 0; i < iters; ++i)
    paged_attention_decode_kernel<<<grid, block>>>(
        d_q, d_k, d_v, d_bt, d_sl, d_out, scale,
        num_q_heads, num_kv_heads, max_num_blocks);
  HIP_CHECK(hipEventRecord(t1));
  HIP_CHECK(hipEventSynchronize(t1));
  float ms = 0.0f;
  HIP_CHECK(hipEventElapsedTime(&ms, t0, t1));

  std::vector<float> out_gpu(q.size());
  HIP_CHECK(hipMemcpy(out_gpu.data(), d_out, out_gpu.size() * sizeof(float), hipMemcpyDeviceToHost));

  // CPU reference
  std::vector<float> out_cpu(q.size());
  cpu_reference(q, k_cache, v_cache, block_table, seq_lens, out_cpu, scale,
                num_seqs, num_q_heads, num_kv_heads, max_num_blocks);

  // Compare
  float max_abs_err = 0.0f;
  for (size_t i = 0; i < out_cpu.size(); ++i)
    max_abs_err = std::max(max_abs_err, std::fabs(out_cpu[i] - out_gpu[i]));

  printf("paged-attention decode (fp32, portable HIP, gfx1201)\n");
  printf("  num_seqs=%d  num_q_heads=%d  num_kv_heads=%d  GROUP=%d\n",
         num_seqs, num_q_heads, num_kv_heads, GROUP);
  printf("  HEAD_DIM=%d  BLOCK_SIZE=%d  seq_lens={40,17,64}\n", HEAD_DIM, BLOCK_SIZE);
  printf("  kernel time: %.4f ms/iter (avg of %d)\n", ms / iters, iters);
  printf("  max abs error vs CPU: %.3e\n", max_abs_err);

  const bool pass = max_abs_err < 1e-4f;
  printf("%s\n", pass ? "PASS" : "FAIL");

  HIP_CHECK(hipFree(d_q));  HIP_CHECK(hipFree(d_k));  HIP_CHECK(hipFree(d_v));
  HIP_CHECK(hipFree(d_out)); HIP_CHECK(hipFree(d_bt)); HIP_CHECK(hipFree(d_sl));
  return pass ? 0 : 1;
}
