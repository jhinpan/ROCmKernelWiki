#include <hip/hip_runtime.h>

using i32x8 = int __attribute__((ext_vector_type(8)));
using f32x4 = float __attribute__((ext_vector_type(4)));

extern "C" __global__ void f8f6f4_probe(const i32x8* first,
                                         const i32x8* second,
                                         const int* first_scale,
                                         const int* second_scale,
                                         f32x4* output)
{
    const unsigned int lane = threadIdx.x & 63u;
    const f32x4 accumulator = {0.0f, 0.0f, 0.0f, 0.0f};
    output[lane] =
        __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(
            first[lane],
            second[lane],
            accumulator,
            0,
            0,
            0,
            first_scale[lane],
            0,
            second_scale[lane]);
}
