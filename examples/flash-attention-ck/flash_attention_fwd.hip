// FlashAttention-2 forward — PORTABLE pure-HIP reference (fp32).
//
// This is the *algorithmic* core of the CK-tile FlashAttention pipeline written
// in plain HIP so it BUILDS AND RUNS on RDNA4 (gfx1201), where MFMA does not
// execute. The production AMD path (ck_tile BlockFmhaPipelineQRKSVS) issues the
// two GEMMs as v_mfma_* matrix-core instructions on CDNA (MI300X/MI350); here we
// reproduce the same online-softmax recurrence with scalar FMA math so the
// numerics are demonstrable and self-checkable on consumer hardware.
//
// Mapping: one thread block per (head, query-tile). Each thread owns one query
// row. The block streams over KV tiles of width BC, maintaining the running
// (m, l, O) state in registers exactly as FlashAttention-2 does:
//
//   O = softmax(scale * Q·Kᵀ) · V       (non-causal, single batch, single head set)
//
// Verified against a naive CPU softmax(QK^T)V reference.

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>

#define HIP_CHECK(x)                                                         \
    do {                                                                    \
        hipError_t err = (x);                                              \
        if (err != hipSuccess) {                                           \
            fprintf(stderr, "HIP error %s at %s:%d\n",                     \
                    hipGetErrorString(err), __FILE__, __LINE__);          \
            exit(1);                                                        \
        }                                                                  \
    } while (0)

// Tunable tile sizes (kept small so the example is easy to follow).
constexpr int BR = 64;   // query rows per block  (== blockDim.x, one row/thread)
constexpr int BC = 64;   // key/value columns per streamed KV tile
constexpr int DMAX = 128; // max head dimension supported by the static buffers

// FlashAttention-2 forward kernel.
// Grid:  (numQueryTiles, numHeads)
// Block: BR threads (one per query row in the tile)
// Q: [H, N, D] row-major   K: [H, N, D]   V: [H, N, D]   O: [H, N, D]
__global__ void flash_attn_fwd_kernel(const float* __restrict__ Q,
                                      const float* __restrict__ K,
                                      const float* __restrict__ V,
                                      float* __restrict__ O,
                                      int N, int D, float scale)
{
    const int head    = blockIdx.y;
    const int qtile   = blockIdx.x;
    const int row     = qtile * BR + threadIdx.x;   // global query row
    const int tid     = threadIdx.x;

    const float* Qh = Q + (size_t)head * N * D;
    const float* Kh = K + (size_t)head * N * D;
    const float* Vh = V + (size_t)head * N * D;
    float*       Oh = O + (size_t)head * N * D;

    // Per-thread (per-row) FlashAttention-2 state, kept in registers.
    float qreg[DMAX];      // this row of Q
    float oacc[DMAX];      // output accumulator O_i
    float m_i = -INFINITY; // running row max
    float l_i = 0.0f;      // running row sum

    const bool active = (row < N);
    for (int d = 0; d < D; ++d) {
        qreg[d] = active ? Qh[row * D + d] : 0.0f;
        oacc[d] = 0.0f;
    }

    // Shared tiles for the current KV block: K_j (BC x D) and V_j (BC x D).
    __shared__ float Ks[BC * DMAX];
    __shared__ float Vs[BC * DMAX];

    for (int kv0 = 0; kv0 < N; kv0 += BC) {
        const int bc = min(BC, N - kv0);

        // Cooperative load of this KV tile into LDS (all threads help).
        for (int idx = tid; idx < bc * D; idx += BR) {
            int r = idx / D, c = idx % D;
            Ks[r * DMAX + c] = Kh[(kv0 + r) * D + c];
            Vs[r * DMAX + c] = Vh[(kv0 + r) * D + c];
        }
        __syncthreads();

        if (active) {
            // 1) scores s_j = scale * (q · k_j) for the BC keys in this tile,
            //    and tile-local row max.
            float s[BC];
            float m_tile = -INFINITY;
            for (int j = 0; j < bc; ++j) {
                float acc = 0.0f;
                for (int d = 0; d < D; ++d)
                    acc += qreg[d] * Ks[j * DMAX + d];
                acc *= scale;
                s[j] = acc;
                m_tile = fmaxf(m_tile, acc);
            }

            // 2) online-softmax update: new running max, rescale old state.
            float m_new = fmaxf(m_i, m_tile);
            float alpha = __expf(m_i - m_new);   // rescale factor for old O,l

            float l_tile = 0.0f;
            for (int j = 0; j < bc; ++j) {
                float p = __expf(s[j] - m_new);  // P_ij
                s[j]    = p;
                l_tile += p;
            }

            // 3) O_i = alpha*O_i + P_ij · V_j   ;   l_i = alpha*l_i + sum(P)
            for (int d = 0; d < D; ++d) {
                float pv = 0.0f;
                for (int j = 0; j < bc; ++j)
                    pv += s[j] * Vs[j * DMAX + d];
                oacc[d] = alpha * oacc[d] + pv;
            }
            l_i = alpha * l_i + l_tile;
            m_i = m_new;
        }
        __syncthreads(); // protect LDS reuse before next tile load
    }

    // 4) final normalization (the FA-2 hoisted 1/l division) and writeback.
    if (active) {
        float inv = 1.0f / l_i;
        for (int d = 0; d < D; ++d)
            Oh[row * D + d] = oacc[d] * inv;
    }
}

// ---- naive CPU reference: O = softmax(scale * Q·Kᵀ) · V -----------------------
static void cpu_reference(const std::vector<float>& Q, const std::vector<float>& K,
                          const std::vector<float>& V, std::vector<float>& O,
                          int H, int N, int D, float scale)
{
    std::vector<float> s(N);
    for (int h = 0; h < H; ++h) {
        const float* Qh = Q.data() + (size_t)h * N * D;
        const float* Kh = K.data() + (size_t)h * N * D;
        const float* Vh = V.data() + (size_t)h * N * D;
        float*       Oh = O.data() + (size_t)h * N * D;
        for (int i = 0; i < N; ++i) {
            float m = -INFINITY;
            for (int j = 0; j < N; ++j) {
                float acc = 0.0f;
                for (int d = 0; d < D; ++d) acc += Qh[i * D + d] * Kh[j * D + d];
                acc *= scale;
                s[j] = acc;
                m = std::fmax(m, acc);
            }
            float l = 0.0f;
            for (int j = 0; j < N; ++j) { s[j] = std::exp(s[j] - m); l += s[j]; }
            for (int d = 0; d < D; ++d) {
                float o = 0.0f;
                for (int j = 0; j < N; ++j) o += s[j] * Vh[j * D + d];
                Oh[i * D + d] = o / l;
            }
        }
    }
}

int main(int argc, char** argv)
{
    int H = 4;    // heads
    int N = 256;  // sequence length
    int D = 64;   // head dimension (<= DMAX)
    if (argc > 1) H = atoi(argv[1]);
    if (argc > 2) N = atoi(argv[2]);
    if (argc > 3) D = atoi(argv[3]);
    if (D > DMAX) { fprintf(stderr, "D must be <= %d\n", DMAX); return 1; }

    const float scale = 1.0f / std::sqrt((float)D);
    const size_t elems = (size_t)H * N * D;

    std::mt19937 rng(1234);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<float> hQ(elems), hK(elems), hV(elems), hO(elems), hRef(elems);
    for (size_t i = 0; i < elems; ++i) { hQ[i] = dist(rng); hK[i] = dist(rng); hV[i] = dist(rng); }

    float *dQ, *dK, *dV, *dO;
    HIP_CHECK(hipMalloc(&dQ, elems * sizeof(float)));
    HIP_CHECK(hipMalloc(&dK, elems * sizeof(float)));
    HIP_CHECK(hipMalloc(&dV, elems * sizeof(float)));
    HIP_CHECK(hipMalloc(&dO, elems * sizeof(float)));
    HIP_CHECK(hipMemcpy(dQ, hQ.data(), elems * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dK, hK.data(), elems * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dV, hV.data(), elems * sizeof(float), hipMemcpyHostToDevice));

    dim3 block(BR);
    dim3 grid((N + BR - 1) / BR, H);

    // warmup
    flash_attn_fwd_kernel<<<grid, block>>>(dQ, dK, dV, dO, N, D, scale);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    // timed runs
    const int iters = 50;
    hipEvent_t t0, t1;
    HIP_CHECK(hipEventCreate(&t0));
    HIP_CHECK(hipEventCreate(&t1));
    HIP_CHECK(hipEventRecord(t0));
    for (int it = 0; it < iters; ++it)
        flash_attn_fwd_kernel<<<grid, block>>>(dQ, dK, dV, dO, N, D, scale);
    HIP_CHECK(hipEventRecord(t1));
    HIP_CHECK(hipEventSynchronize(t1));
    float ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&ms, t0, t1));
    ms /= iters;

    HIP_CHECK(hipMemcpy(hO.data(), dO, elems * sizeof(float), hipMemcpyDeviceToHost));

    cpu_reference(hQ, hK, hV, hRef, H, N, D, scale);

    double max_abs = 0.0, max_rel = 0.0;
    for (size_t i = 0; i < elems; ++i) {
        double a = hO[i], b = hRef[i];
        double ae = std::fabs(a - b);
        double re = ae / (std::fabs(b) + 1e-6);
        max_abs = std::fmax(max_abs, ae);
        max_rel = std::fmax(max_rel, re);
    }

    // FLOPs: 2 GEMMs of N*N*D each (QK^T and PV), per head: 2 * 2*N*N*D.
    double flops = (double)H * 2.0 * 2.0 * (double)N * N * D;
    double gflops = flops / (ms * 1e-3) / 1e9;

    printf("FlashAttention-2 fwd (portable HIP, fp32) on gfx1201\n");
    printf("  H=%d N=%d D=%d  BR=%d BC=%d  scale=%.5f\n", H, N, D, BR, BC, scale);
    printf("  avg kernel time: %.4f ms   (%.1f GFLOP/s)\n", ms, gflops);
    printf("  max abs error: %.3e   max rel error: %.3e\n", max_abs, max_rel);

    const double tol = 1e-3;
    bool pass = max_abs < tol;
    printf("%s\n", pass ? "PASS" : "FAIL");

    HIP_CHECK(hipFree(dQ)); HIP_CHECK(hipFree(dK));
    HIP_CHECK(hipFree(dV)); HIP_CHECK(hipFree(dO));
    return pass ? 0 : 1;
}
