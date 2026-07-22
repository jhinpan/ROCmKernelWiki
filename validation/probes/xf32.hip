#include <hip/hip_runtime.h>

using f32x2 = float __attribute__((ext_vector_type(2)));
using f32x4 = float __attribute__((ext_vector_type(4)));

extern "C" __global__ void xf32_probe(const f32x2* first,
                                      const f32x2* second,
                                      f32x4* output)
{
    const unsigned int lane = threadIdx.x & 63u;
    const f32x4 accumulator = {0.0f, 0.0f, 0.0f, 0.0f};
    output[lane] = __builtin_amdgcn_mfma_f32_16x16x8_xf32(
        first[lane], second[lane], accumulator, 0, 0, 0);
}
