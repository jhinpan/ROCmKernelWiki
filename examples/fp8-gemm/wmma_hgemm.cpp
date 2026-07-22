// wmma_hgemm.cpp — portable rocWMMA API FP16 GEMM, verified on gfx950.
//
// This is the runnable fallback for the FP8 GEMM page. It shows the same tiled
// matrix-core structure in FP16 through rocWMMA; gfx950 emits MFMA instructions.
//
// D = A (MxK, row-major) * B (KxN, col-major) -> C (MxN, row-major), FP32 accum.
// Each wave computes one 16x16 output tile. Self-checks against a CPU reference.
//
// Build (runs on gfx950):
//   hipcc --offload-arch=gfx950 -I/opt/rocm/include wmma_hgemm.cpp -o wmma_hgemm

#include <hip/hip_runtime.h>
#include <rocwmma/rocwmma.hpp>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

using namespace rocwmma;

constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

#define HIP_CHECK(cmd)                                                        \
    do {                                                                     \
        hipError_t e = (cmd);                                                \
        if (e != hipSuccess) {                                              \
            fprintf(stderr, "HIP error %s at %s:%d\n",                       \
                    hipGetErrorString(e), __FILE__, __LINE__);             \
            exit(1);                                                         \
        }                                                                   \
    } while (0)

// A: MxK row-major (lda=K). B: KxN col-major (ldb=K). C: MxN row-major (ldc=N).
__global__ void wmma_hgemm(const __half* A, const __half* B, float* C,
                           int M, int N, int K)
{
    // One wave per 16x16 output tile.
    int tileM = blockIdx.y;       // tile row
    int tileN = blockIdx.x;       // tile col

    auto fragA = fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, row_major>();
    auto fragB = fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, col_major>();
    auto fragC = fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float>();
    fill_fragment(fragC, 0.0f);

    for (int k = 0; k < K; k += WMMA_K) {
        const __half* aptr = A + (tileM * WMMA_M) * K + k;     // row-major, lda=K
        const __half* bptr = B + (tileN * WMMA_N) * K + k;     // col-major, ldb=K
        load_matrix_sync(fragA, aptr, K);
        load_matrix_sync(fragB, bptr, K);
        mma_sync(fragC, fragA, fragB, fragC);
    }

    float* cptr = C + (tileM * WMMA_M) * N + (tileN * WMMA_N);
    store_matrix_sync(cptr, fragC, N, mem_row_major);
}

int main()
{
    // Dimensions are multiples of the 16x16x16 tile.
    const int M = 256, N = 256, K = 256;

    std::vector<__half> hA(M * K), hB(K * N);
    std::vector<float>  hC(M * N), ref(M * N);

    srand(1234);
    auto rf = []() { return (float)(rand() % 7 - 3) * 0.25f; }; // small ints/4
    for (auto& x : hA) x = __float2half(rf());
    for (auto& x : hB) x = __float2half(rf());

    // CPU reference: A row-major (MxK), B col-major (KxN) -> C row-major.
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j) {
            float acc = 0.f;
            for (int k = 0; k < K; ++k)
                acc += __half2float(hA[i * K + k]) * __half2float(hB[j * K + k]);
            ref[i * N + j] = acc;
        }

    __half *dA, *dB; float* dC;
    HIP_CHECK(hipMalloc(&dA, hA.size() * sizeof(__half)));
    HIP_CHECK(hipMalloc(&dB, hB.size() * sizeof(__half)));
    HIP_CHECK(hipMalloc(&dC, hC.size() * sizeof(float)));
    HIP_CHECK(hipMemcpy(dA, hA.data(), hA.size() * sizeof(__half), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dB, hB.data(), hB.size() * sizeof(__half), hipMemcpyHostToDevice));

    // rocWMMA needs one full wave per 16x16 tile. Query the target's wave size.
    int warp = 0;
    HIP_CHECK(hipDeviceGetAttribute(&warp, hipDeviceAttributeWarpSize, 0));
    dim3 block(warp, 1, 1);
    dim3 grid(N / WMMA_N, M / WMMA_M, 1);

    // Warmup + timed run.
    hipLaunchKernelGGL(wmma_hgemm, grid, block, 0, 0, dA, dB, dC, M, N, K);
    HIP_CHECK(hipDeviceSynchronize());

    hipEvent_t t0, t1;
    HIP_CHECK(hipEventCreate(&t0));
    HIP_CHECK(hipEventCreate(&t1));
    const int iters = 50;
    HIP_CHECK(hipEventRecord(t0));
    for (int it = 0; it < iters; ++it)
        hipLaunchKernelGGL(wmma_hgemm, grid, block, 0, 0, dA, dB, dC, M, N, K);
    HIP_CHECK(hipEventRecord(t1));
    HIP_CHECK(hipEventSynchronize(t1));
    float ms = 0.f;
    HIP_CHECK(hipEventElapsedTime(&ms, t0, t1));
    double avg_ms = ms / iters;
    double gflops = 2.0 * M * N * K / (avg_ms * 1e-3) / 1e9;

    HIP_CHECK(hipMemcpy(hC.data(), dC, hC.size() * sizeof(float), hipMemcpyDeviceToHost));

    double maxerr = 0.0;
    for (int i = 0; i < M * N; ++i)
        maxerr = fmax(maxerr, fabs((double)hC[i] - (double)ref[i]));

    printf("rocWMMA FP16 GEMM  M=%d N=%d K=%d (warpSize=%d)\n", M, N, K, warp);
    printf("avg %.4f ms/iter   %.1f GFLOP/s\n", avg_ms, gflops);
    printf("max abs error = %.6f\n", maxerr);
    bool pass = maxerr < 1e-2;   // FP16 inputs, exact-ish small values
    printf("%s\n", pass ? "PASS" : "FAIL");

    HIP_CHECK(hipFree(dA));
    HIP_CHECK(hipFree(dB));
    HIP_CHECK(hipFree(dC));
    return pass ? 0 : 1;
}
