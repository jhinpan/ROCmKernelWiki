#include <hip/hip_runtime.h>

extern "C" __global__ void permlane16_swap_probe(const int* first,
                                                  const int* second,
                                                  int* output)
{
    const unsigned int lane = threadIdx.x & 63u;
    const auto swapped = __builtin_amdgcn_permlane16_swap(
        first[lane], second[lane], false, false);
    output[2 * lane] = swapped[0];
    output[2 * lane + 1] = swapped[1];
}
