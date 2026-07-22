// Portable rocWMMA FP16 GEMM: C = A * B
//   A: MxK row-major (fp16)
//   B: KxN col-major (fp16)   [rocWMMA matrix_b wants col_major here]
//   C: MxN row-major (fp32)
// Each gfx950 wave computes one 16x16 output tile, accumulating over K in
// 16x16x16 fragments. The source uses rocWMMA's API; the gfx950 build emits
// MFMA instructions.
//
// Verifies against a CPU reference (fp16 inputs, fp32 accumulate) and prints
// PASS/FAIL with max abs/relative error and a rough TFLOPS number.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <rocwmma/rocwmma.hpp>

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>

using namespace rocwmma;

#define HIP_CHECK(cmd)                                                          \
    do {                                                                       \
        hipError_t e = (cmd);                                                  \
        if (e != hipSuccess) {                                                 \
            fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(e),   \
                    __FILE__, __LINE__);                                       \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)

constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

// One wave per 16x16 output tile.
// blockDim.x must equal the wave size; blockDim.y waves stack along the M axis.
__global__ void hgemm_wmma_kernel(const __half* __restrict__ A,  // MxK row-major
                                  const __half* __restrict__ B,  // KxN col-major
                                  float* __restrict__ C,         // MxN row-major
                                  int M, int N, int K)
{
    // Which 16x16 output tile does this wave own?
    int warpsPerBlockX = blockDim.x / warpSize;
    int warpIdInBlock  = threadIdx.x / warpSize;
    int tileCol = (blockIdx.x * warpsPerBlockX + warpIdInBlock); // along N
    int tileRow = (blockIdx.y * blockDim.y + threadIdx.y);       // along M

    int rowStart = tileRow * WMMA_M;
    int colStart = tileCol * WMMA_N;
    if (rowStart >= M || colStart >= N) return;

    auto fragC = fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float>();
    fill_fragment(fragC, 0.0f);

    for (int k = 0; k < K; k += WMMA_K) {
        auto fragA = fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, row_major>();
        auto fragB = fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, col_major>();

        // A tile: rows [rowStart, +16), cols [k, +16). Row-major lda = K.
        const __half* aPtr = A + rowStart * K + k;
        // B tile (col-major, ldb = K): cols [colStart,+16), rows [k,+16).
        const __half* bPtr = B + colStart * K + k;

        load_matrix_sync(fragA, aPtr, K);
        load_matrix_sync(fragB, bPtr, K);
        mma_sync(fragC, fragA, fragB, fragC);
    }

    // Store: C row-major, ldc = N.
    float* cPtr = C + rowStart * N + colStart;
    store_matrix_sync(cPtr, fragC, N, mem_row_major);
}

int main(int argc, char** argv)
{
    int M = 256, N = 256, K = 256;
    if (argc == 4) { M = atoi(argv[1]); N = atoi(argv[2]); K = atoi(argv[3]); }
    // rocWMMA tile path requires multiples of 16.
    auto roundUp = [](int x) { return ((x + 15) / 16) * 16; };
    M = roundUp(M); N = roundUp(N); K = roundUp(K);
    printf("rocWMMA FP16 GEMM  M=%d N=%d K=%d  (16x16x16 fragments)\n", M, N, K);

    int dev = 0; hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, dev));
    printf("Device: %s (%s), warpSize=%d\n", prop.name, prop.gcnArchName, prop.warpSize);

    std::vector<__half> hA(size_t(M) * K), hB(size_t(K) * N);
    std::vector<float>  hC(size_t(M) * N, 0.0f), hRef(size_t(M) * N, 0.0f);

    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto& x : hA) x = __float2half(dist(rng));
    for (auto& x : hB) x = __float2half(dist(rng));

    // CPU reference: fp16 in, fp32 accumulate. B is col-major (ldb = K).
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j) {
            float acc = 0.0f;
            for (int k = 0; k < K; ++k)
                acc += __half2float(hA[size_t(i) * K + k]) *
                       __half2float(hB[size_t(j) * K + k]);
            hRef[size_t(i) * N + j] = acc;
        }

    __half *dA, *dB; float* dC;
    HIP_CHECK(hipMalloc(&dA, hA.size() * sizeof(__half)));
    HIP_CHECK(hipMalloc(&dB, hB.size() * sizeof(__half)));
    HIP_CHECK(hipMalloc(&dC, hC.size() * sizeof(float)));
    HIP_CHECK(hipMemcpy(dA, hA.data(), hA.size() * sizeof(__half), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dB, hB.data(), hB.size() * sizeof(__half), hipMemcpyHostToDevice));

    const int waveSize = prop.warpSize;
    const int warpsX = 4;             // 4 waves across N per block
    const int warpsY = 4;             // 4 tile-rows across M per block
    dim3 block(waveSize * warpsX, warpsY);
    dim3 grid((N / WMMA_N + warpsX - 1) / warpsX,
              (M / WMMA_M + warpsY - 1) / warpsY);

    // Warm-up + correctness run.
    hipLaunchKernelGGL(hgemm_wmma_kernel, grid, block, 0, 0, dA, dB, dC, M, N, K);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(hC.data(), dC, hC.size() * sizeof(float), hipMemcpyDeviceToHost));

    // Timed runs.
    const int iters = 50;
    hipEvent_t t0, t1;
    HIP_CHECK(hipEventCreate(&t0)); HIP_CHECK(hipEventCreate(&t1));
    HIP_CHECK(hipEventRecord(t0));
    for (int it = 0; it < iters; ++it)
        hipLaunchKernelGGL(hgemm_wmma_kernel, grid, block, 0, 0, dA, dB, dC, M, N, K);
    HIP_CHECK(hipEventRecord(t1));
    HIP_CHECK(hipEventSynchronize(t1));
    float ms = 0.0f; HIP_CHECK(hipEventElapsedTime(&ms, t0, t1));
    double avgMs = ms / iters;
    double tflops = (2.0 * M * N * K) / (avgMs * 1e-3) / 1e12;

    // Verify.
    double maxAbs = 0.0, maxRel = 0.0;
    for (size_t i = 0; i < hC.size(); ++i) {
        double diff = std::fabs(double(hC[i]) - double(hRef[i]));
        double rel  = diff / (std::fabs(double(hRef[i])) + 1e-6);
        maxAbs = std::max(maxAbs, diff);
        maxRel = std::max(maxRel, rel);
    }
    // fp16 inputs: tolerance scales with K.
    double tol = 0.04 * K;
    printf("max abs err = %.4f   max rel err = %.5f   (tol abs = %.2f)\n",
           maxAbs, maxRel, tol);
    printf("avg kernel time = %.4f ms   ~%.1f GFLOP/s (%.3f TFLOPS)\n",
           avgMs, tflops * 1e3, tflops);

    bool pass = (maxAbs <= tol);
    printf("%s\n", pass ? "PASS" : "FAIL");

    HIP_CHECK(hipFree(dA)); HIP_CHECK(hipFree(dB)); HIP_CHECK(hipFree(dC));
    HIP_CHECK(hipEventDestroy(t0)); HIP_CHECK(hipEventDestroy(t1));
    return pass ? 0 : 1;
}
