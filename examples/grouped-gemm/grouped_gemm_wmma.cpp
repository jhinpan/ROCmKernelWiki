// grouped_gemm_wmma.cpp
//
// PORTABLE rocWMMA grouped GEMM: several INDEPENDENT (Mi, Ni, Ki) problems are
// dispatched in ONE kernel launch via a flattened tile -> group lookup.
//
//   for g in 0..G-1:  C_g[Mg, Ng] = A_g[Mg, Kg] * B_g[Kg, Ng]
//
// The defining feature vs a batched GEMM is that Mg, Ng and Kg differ per group
// (the shape an MoE router produces). We build a per-group descriptor table and a
// prefix sum over each group's 16x16 output-tile count; the kernel launches
// total_tiles blocks (one wave each) and every block binary/linear-searches the
// prefix sum to discover which group + (m_tile, n_tile) it owns. One launch
// covers all groups regardless of how uneven the sizes are.
//
// fp16 input  ->  fp32 accumulate  ->  fp32 output.
// The source uses rocWMMA's 16x16x16 fragment API; on gfx950 the compiler emits
// MFMA instructions. This file runs on gfx950 and self-checks each group against
// a CPU reference.
//
// Ragged sizes are handled by padding each matrix's storage to a multiple of 16
// with ZEROS, so rocWMMA's (non-bounds-checked) loads are always in range and
// the padding contributes 0 to the dot product. The CPU reference and the PASS
// check use the LOGICAL (unpadded) Mg/Ng/Kg.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <rocwmma/rocwmma.hpp>

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <random>

using namespace rocwmma;

#define HIP_CHECK(cmd)                                                         \
  do {                                                                         \
    hipError_t e = (cmd);                                                      \
    if (e != hipSuccess) {                                                     \
      fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(e),        \
              __FILE__, __LINE__);                                            \
      std::exit(1);                                                           \
    }                                                                          \
  } while (0)

static constexpr int WMMA_M = 16;
static constexpr int WMMA_N = 16;
static constexpr int WMMA_K = 16;

// Per-group problem descriptor (lives in device memory).
struct GroupDesc {
  int    M, N, K;             // logical sizes
  int    lda, ldb, ldc;       // padded leading dims (multiples of 16)
  size_t aoff, boff, coff;    // element offsets into the flat A/B/C buffers
  int    ntiles;              // ceil(N/16): output tiles along N for this group
  int    tile_base;           // first flat tile id owned by this group
};

// One launch, all groups. grid.x = total_tiles, blockDim.x = one wave.
__global__ void grouped_gemm_kernel(const GroupDesc* __restrict__ groups,
                                    int G, int total_tiles,
                                    const __half* __restrict__ A,
                                    const __half* __restrict__ B,
                                    float* __restrict__ C) {
  int tile_id = blockIdx.x;
  if (tile_id >= total_tiles) return;

  // Which group owns this flat tile? (G is small -> linear scan is fine.)
  int g = 0;
  for (int i = G - 1; i >= 0; --i) {
    if (tile_id >= groups[i].tile_base) { g = i; break; }
  }
  const GroupDesc d = groups[g];

  int local  = tile_id - d.tile_base;
  int m_tile = local / d.ntiles;
  int n_tile = local % d.ntiles;

  const __half* Ag = A + d.aoff;   // M x K row-major, lda = padded K
  const __half* Bg = B + d.boff;   // K x N col-major, ldb = padded K
  float*        Cg = C + d.coff;   // M x N row-major, ldc = padded N

  auto fragC = fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float>();
  fill_fragment(fragC, 0.0f);

  int Kp = d.lda;  // padded K (row-major A leading dim == padded K)
  for (int k0 = 0; k0 < Kp; k0 += WMMA_K) {
    auto fragA = fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, row_major>();
    auto fragB = fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, col_major>();

    const __half* aptr = Ag + (size_t)(m_tile * WMMA_M) * d.lda + k0;
    const __half* bptr = Bg + (size_t)(n_tile * WMMA_N) * d.ldb + k0;

    load_matrix_sync(fragA, aptr, d.lda);
    load_matrix_sync(fragB, bptr, d.ldb);
    mma_sync(fragC, fragA, fragB, fragC);
  }

  float* cptr = Cg + (size_t)(m_tile * WMMA_M) * d.ldc + (n_tile * WMMA_N);
  store_matrix_sync(cptr, fragC, d.ldc, mem_row_major);
}

static int round16(int x) { return (x + 15) / 16 * 16; }

int main() {
  // Independent problems with deliberately uneven, non-16-multiple sizes.
  struct Prob { int M, N, K; };
  std::vector<Prob> probs = {
      { 64,  48,  80},
      { 32,  96,  32},
      {128,  16,  64},
      { 16, 128, 112},
      { 80,  80,  48},
      { 17,  33,  49},   // fully ragged: none divisible by 16
  };
  const int G = (int)probs.size();

  std::mt19937 rng(1234);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

  // Build padded host buffers (zero-initialised so padding contributes 0).
  std::vector<GroupDesc> hdesc(G);
  std::vector<size_t> aSz(G), bSz(G), cSz(G);
  size_t aTot = 0, bTot = 0, cTot = 0;
  int tile_base = 0;
  for (int g = 0; g < G; ++g) {
    int M = probs[g].M, N = probs[g].N, K = probs[g].K;
    int Mp = round16(M), Np = round16(N), Kp = round16(K);
    GroupDesc d{};
    d.M = M; d.N = N; d.K = K;
    d.lda = Kp; d.ldb = Kp; d.ldc = Np;   // A row-major(MxK), B col-major(KxN), C row-major(MxN)
    d.aoff = aTot; d.boff = bTot; d.coff = cTot;
    int mtiles = Mp / 16, ntiles = Np / 16;
    d.ntiles = ntiles;
    d.tile_base = tile_base;
    tile_base += mtiles * ntiles;
    hdesc[g] = d;
    aSz[g] = (size_t)Mp * Kp;
    bSz[g] = (size_t)Kp * Np;
    cSz[g] = (size_t)Mp * Np;
    aTot += aSz[g]; bTot += bSz[g]; cTot += cSz[g];
  }
  const int total_tiles = tile_base;

  std::vector<__half> hA(aTot, __float2half(0.0f));
  std::vector<__half> hB(bTot, __float2half(0.0f));
  std::vector<float>  hC(cTot, 0.0f);
  // Reference values stored in float for the CPU check.
  std::vector<float> refA(aTot, 0.0f), refB(bTot, 0.0f);

  for (int g = 0; g < G; ++g) {
    const GroupDesc& d = hdesc[g];
    // A: M x K row-major, padded leading dim lda.
    for (int i = 0; i < d.M; ++i)
      for (int k = 0; k < d.K; ++k) {
        float v = dist(rng);
        size_t idx = d.aoff + (size_t)i * d.lda + k;
        hA[idx] = __float2half(v);
        refA[idx] = __half2float(__float2half(v));  // round-trip to match device
      }
    // B: K x N col-major, padded leading dim ldb.
    for (int k = 0; k < d.K; ++k)
      for (int j = 0; j < d.N; ++j) {
        float v = dist(rng);
        size_t idx = d.boff + (size_t)j * d.ldb + k;
        hB[idx] = __float2half(v);
        refB[idx] = __half2float(__float2half(v));
      }
  }

  // Device buffers.
  GroupDesc* dDesc; __half *dA, *dB; float* dC;
  HIP_CHECK(hipMalloc(&dDesc, G * sizeof(GroupDesc)));
  HIP_CHECK(hipMalloc(&dA, aTot * sizeof(__half)));
  HIP_CHECK(hipMalloc(&dB, bTot * sizeof(__half)));
  HIP_CHECK(hipMalloc(&dC, cTot * sizeof(float)));
  HIP_CHECK(hipMemcpy(dDesc, hdesc.data(), G * sizeof(GroupDesc), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(dA, hA.data(), aTot * sizeof(__half), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(dB, hB.data(), bTot * sizeof(__half), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemset(dC, 0, cTot * sizeof(float)));

  // One wave per 16x16 output tile.
  hipDeviceProp_t prop;
  HIP_CHECK(hipGetDeviceProperties(&prop, 0));
  int wave = prop.warpSize;  // 64 on the verified gfx950 target
  printf("Device: %s  warpSize=%d\n", prop.name, wave);
  printf("Groups: %d   total 16x16 output tiles (one launch): %d\n", G, total_tiles);

  dim3 grid(total_tiles), block(wave);

  // Timed run.
  hipEvent_t t0, t1; HIP_CHECK(hipEventCreate(&t0)); HIP_CHECK(hipEventCreate(&t1));
  // warmup
  grouped_gemm_kernel<<<grid, block>>>(dDesc, G, total_tiles, dA, dB, dC);
  HIP_CHECK(hipGetLastError());
  HIP_CHECK(hipDeviceSynchronize());

  const int iters = 100;
  HIP_CHECK(hipEventRecord(t0));
  for (int it = 0; it < iters; ++it)
    grouped_gemm_kernel<<<grid, block>>>(dDesc, G, total_tiles, dA, dB, dC);
  HIP_CHECK(hipEventRecord(t1));
  HIP_CHECK(hipEventSynchronize(t1));
  float ms = 0.f; HIP_CHECK(hipEventElapsedTime(&ms, t0, t1));
  ms /= iters;

  HIP_CHECK(hipMemcpy(hC.data(), dC, cTot * sizeof(float), hipMemcpyDeviceToHost));

  // CPU reference + per-group check.
  double max_abs_err = 0.0;
  bool all_pass = true;
  double total_flop = 0.0;
  for (int g = 0; g < G; ++g) {
    const GroupDesc& d = hdesc[g];
    double gmax = 0.0;
    for (int i = 0; i < d.M; ++i) {
      for (int j = 0; j < d.N; ++j) {
        float acc = 0.0f;
        for (int k = 0; k < d.K; ++k) {
          float a = refA[d.aoff + (size_t)i * d.lda + k];
          float b = refB[d.boff + (size_t)j * d.ldb + k];
          acc += a * b;
        }
        float got = hC[d.coff + (size_t)i * d.ldc + j];
        double e = std::fabs((double)got - (double)acc);
        if (e > gmax) gmax = e;
      }
    }
    total_flop += 2.0 * d.M * d.N * d.K;
    bool pass = gmax < 1e-1;  // fp16 inputs -> generous tol for K up to 112
    all_pass &= pass;
    if (gmax > max_abs_err) max_abs_err = gmax;
    printf("  group %d  M=%3d N=%3d K=%3d  tiles=%2d  max|err|=%.4e  %s\n",
           g, d.M, d.N, d.K, (round16(d.M)/16)*d.ntiles, gmax, pass ? "ok" : "BAD");
  }

  double gflops = (total_flop / (ms * 1e-3)) / 1e9;
  printf("Avg kernel time: %.4f ms  (%.1f GFLOP/s aggregate over all groups)\n", ms, gflops);
  printf("Overall max abs error: %.4e\n", max_abs_err);
  printf("%s\n", all_pass ? "PASS" : "FAIL");

  HIP_CHECK(hipFree(dDesc)); HIP_CHECK(hipFree(dA));
  HIP_CHECK(hipFree(dB)); HIP_CHECK(hipFree(dC));
  return all_pass ? 0 : 1;
}
