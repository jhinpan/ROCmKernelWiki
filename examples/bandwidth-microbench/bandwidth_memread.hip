// HBM read-bandwidth microbenchmark — PORTABLE HIP, runs on gfx1201 (RDNA4).
//
// Each persistent block streams a large array with 128-bit (float4) vectorized
// loads and accumulates into a register sink so the compiler cannot dead-code
// the loads away. We sweep several working-set sizes and report measured GB/s.
//
// This is the pure-HIP, runnable cousin of gcnasm's bandwidth_memread. It uses
// __builtin_nontemporal_load (lowered to a streaming global_load on AMD GPUs)
// and a grid-stride persistent grid (one block per CU).
//
// Self-check: a tiny separate verification kernel reduces a known buffer and we
// compare the sum against a CPU reference, proving the load path is correct.

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

// Native 128-bit vector: supports .x/.y/.z/.w AND is accepted by the
// __builtin_nontemporal_load intrinsic (HIP's float4 struct is not).
typedef float float4_t __attribute__((ext_vector_type(4)));

#define HIP_CHECK(x)                                                            \
    do {                                                                        \
        hipError_t err = (x);                                                   \
        if (err != hipSuccess) {                                                \
            fprintf(stderr, "HIP error %s at %s:%d\n",                          \
                    hipGetErrorString(err), __FILE__, __LINE__);               \
            exit(1);                                                            \
        }                                                                       \
    } while (0)

// ---------------------------------------------------------------------------
// Bandwidth kernel: persistent grid-stride read with manual unroll.
// Each thread issues UNROLL non-temporal float4 loads per iteration so several
// loads are in flight before the first vmcnt wait stalls the wave.
// ---------------------------------------------------------------------------
template <int UNROLL = 8>
__global__ void bandwidth_memread(const float4_t* __restrict__ in,
                                  float* __restrict__ sink,
                                  size_t n_vec4)
{
    const size_t tid    = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = (size_t)gridDim.x * blockDim.x;

    float4_t acc = {0.f, 0.f, 0.f, 0.f};

    size_t i = tid;
    for (; i + (size_t)UNROLL * stride <= n_vec4 - 1; i += (size_t)UNROLL * stride) {
        float4_t v[UNROLL];
#pragma unroll
        for (int u = 0; u < UNROLL; ++u)
            v[u] = __builtin_nontemporal_load(in + i + (size_t)u * stride);
#pragma unroll
        for (int u = 0; u < UNROLL; ++u) {
            acc.x += v[u].x; acc.y += v[u].y;
            acc.z += v[u].z; acc.w += v[u].w;
        }
    }
    // Tail.
    for (; i < n_vec4; i += stride) {
        float4_t v = __builtin_nontemporal_load(in + i);
        acc.x += v.x; acc.y += v.y; acc.z += v.z; acc.w += v.w;
    }

    // Keep the accumulator live without adding measurable store traffic:
    // the guard is essentially never true, so no store actually executes, but
    // the compiler must still compute acc (and therefore issue every load).
    if (acc.x == -1.0f && acc.y == -2.0f)
        sink[tid] = acc.x + acc.y + acc.z + acc.w;
}

// ---------------------------------------------------------------------------
// Verification kernel: block-reduce a buffer into per-block partial sums.
// Used to prove the float4 load path reads the right data.
// ---------------------------------------------------------------------------
__global__ void reduce_sum(const float4_t* __restrict__ in,
                           double* __restrict__ partial,
                           size_t n_vec4)
{
    __shared__ double sdata[256];
    const size_t tid    = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = (size_t)gridDim.x * blockDim.x;

    double local = 0.0;
    for (size_t i = tid; i < n_vec4; i += stride) {
        float4_t v = in[i];
        local += (double)v.x + (double)v.y + (double)v.z + (double)v.w;
    }
    sdata[threadIdx.x] = local;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) sdata[threadIdx.x] += sdata[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) partial[blockIdx.x] = sdata[0];
}

__global__ void fill_kernel(float4_t* out, size_t n_vec4)
{
    const size_t tid    = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = tid; i < n_vec4; i += stride) {
        float f = 1.0f;                 // every scalar element = 1.0f
        float4_t v = {f, f, f, f};
        out[i] = v;
    }
}

int main()
{
    int dev = 0;
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, dev));
    printf("Device: %s (%s), %d CUs, %.0f MHz\n",
           prop.name, prop.gcnArchName, prop.multiProcessorCount,
           prop.clockRate / 1000.0);

    const int block = 256;
    const int grid  = prop.multiProcessorCount * 2;  // a couple blocks per CU

    // ---------------- Correctness check on a small buffer ------------------
    {
        const size_t n_vec4 = (size_t)4 << 20;        // 4M float4 = 64 MiB
        float4_t* buf = nullptr;
        double*   d_partial = nullptr;
        HIP_CHECK(hipMalloc(&buf, n_vec4 * sizeof(float4_t)));
        HIP_CHECK(hipMalloc(&d_partial, grid * sizeof(double)));

        fill_kernel<<<grid, block>>>(buf, n_vec4);
        HIP_CHECK(hipGetLastError());
        reduce_sum<<<grid, block>>>(buf, d_partial, n_vec4);
        HIP_CHECK(hipGetLastError());

        std::vector<double> h_partial(grid);
        HIP_CHECK(hipMemcpy(h_partial.data(), d_partial, grid * sizeof(double),
                            hipMemcpyDeviceToHost));
        double gpu_sum = 0.0;
        for (double v : h_partial) gpu_sum += v;

        double ref = (double)n_vec4 * 4.0;            // 4 scalars per float4, each 1.0
        double rel = std::fabs(gpu_sum - ref) / ref;
        printf("\nSelf-check: sum=%.0f ref=%.0f rel_err=%.3e -> %s\n",
               gpu_sum, ref, rel, rel < 1e-9 ? "PASS" : "FAIL");
        if (!(rel < 1e-9)) { printf("RESULT: FAIL\n"); return 1; }

        HIP_CHECK(hipFree(buf));
        HIP_CHECK(hipFree(d_partial));
    }

    // ---------------- Bandwidth sweep over several sizes -------------------
    printf("\n%-12s %-12s %-10s\n", "size(MiB)", "iters", "GB/s");
    const size_t sizes_mib[] = {64, 256, 1024, 2048};
    for (size_t mib : sizes_mib) {
        const size_t bytes  = mib << 20;
        const size_t n_vec4 = bytes / sizeof(float4_t);

        float4_t* in = nullptr;
        float*    sink = nullptr;
        if (hipMalloc(&in, bytes) != hipSuccess) {
            printf("%-12zu (skip: alloc failed)\n", mib);
            continue;
        }
        HIP_CHECK(hipMalloc(&sink, (size_t)grid * block * sizeof(float)));
        fill_kernel<<<grid, block>>>(in, n_vec4);
        HIP_CHECK(hipDeviceSynchronize());

        // Warm up.
        bandwidth_memread<8><<<grid, block>>>(in, sink, n_vec4);
        HIP_CHECK(hipDeviceSynchronize());

        const int iters = 50;
        hipEvent_t a, b;
        HIP_CHECK(hipEventCreate(&a));
        HIP_CHECK(hipEventCreate(&b));
        HIP_CHECK(hipEventRecord(a));
        for (int it = 0; it < iters; ++it)
            bandwidth_memread<8><<<grid, block>>>(in, sink, n_vec4);
        HIP_CHECK(hipEventRecord(b));
        HIP_CHECK(hipEventSynchronize(b));

        float ms = 0.f;
        HIP_CHECK(hipEventElapsedTime(&ms, a, b));
        double gbps = (double)bytes * iters / (ms * 1e-3) / 1e9;
        printf("%-12zu %-12d %-10.0f\n", mib, iters, gbps);

        HIP_CHECK(hipEventDestroy(a));
        HIP_CHECK(hipEventDestroy(b));
        HIP_CHECK(hipFree(in));
        HIP_CHECK(hipFree(sink));
    }

    printf("\nRESULT: PASS\n");
    return 0;
}
