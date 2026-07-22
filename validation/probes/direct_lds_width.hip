#include <hip/hip_runtime.h>

#ifndef COPY_BYTES
#error "Compile with -DCOPY_BYTES=<width>"
#endif

static_assert(COPY_BYTES > 0, "COPY_BYTES must be positive");

extern "C" __global__ void direct_lds_width_probe(unsigned char* source,
                                                   unsigned char* output)
{
    __shared__ __align__(16) unsigned char tile[64 * COPY_BYTES];
    const unsigned int lane = threadIdx.x & 63u;
    auto* const lds_base =
        (__attribute__((address_space(3))) unsigned char*)tile;

    // Contract under test: source is lane-varying; the LDS destination base is
    // wave-uniform. Hardware adds COPY_BYTES * lane to the LDS destination.
    __builtin_amdgcn_load_to_lds(source + lane * COPY_BYTES,
                                 lds_base,
                                 COPY_BYTES,
                                 0,
                                 0);
    __builtin_amdgcn_s_waitcnt(0);
    __syncthreads();

    output[lane * COPY_BYTES] = tile[lane * COPY_BYTES];
}
