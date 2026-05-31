# Diff summary

- **files changed:** 149
- **lines:** +410 / -404
- **kernel-ish files:** 149

## Files (by churn)

- `profiler/include/profiler/profile_transpose_impl.hpp`  (+178/-178)
- `profiler/src/profile_transpose.cpp`  (+81/-81)
- `profiler/include/profiler/profile_grouped_conv_fwd_outelementop_impl.hpp`  (+3/-0)
- `profiler/src/profile_grouped_conv_fwd_outelementop.cpp`  (+3/-0)
- `profiler/include/profiler/common.hpp`  (+1/-1)
- `profiler/include/profiler/data_type_enum.hpp`  (+1/-1)
- `profiler/include/profiler/profile_avg_pool2d_bwd_impl.hpp`  (+1/-1)
- `profiler/include/profiler/profile_avg_pool3d_bwd_impl.hpp`  (+1/-1)
- `profiler/include/profiler/profile_batched_gemm_add_relu_gemm_add_impl.hpp`  (+1/-1)
- `profiler/include/profiler/profile_batched_gemm_b_scale_impl.hpp`  (+1/-1)
- `profiler/include/profiler/profile_batched_gemm_bias_softmax_gemm_permute_impl.hpp`  (+1/-1)
- `profiler/include/profiler/profile_batched_gemm_gemm_impl.hpp`  (+1/-1)
- `profiler/include/profiler/profile_batched_gemm_impl.hpp`  (+1/-1)
- `profiler/include/profiler/profile_batched_gemm_reduce_impl.hpp`  (+1/-1)
- `profiler/include/profiler/profile_batched_gemm_softmax_gemm_impl.hpp`  (+1/-1)

## Key added lines (kernel files)

**`profiler/include/profiler/profile_transpose_impl.hpp`**
```
namespace ck {
namespace profiler {
template <typename HostTensorA, typename HostTensorB, typename Functor>
void host_elementwise4D(HostTensorB& B_ndhwc, const HostTensorA& A_ncdhw, Functor functor)
```

**`profiler/src/profile_transpose.cpp`**
```
enum struct DataType
F32_F32_F32_F32_F32, // 0
F16_F16_F16_F16_F16, // 1
static void print_helper_msg()
```
