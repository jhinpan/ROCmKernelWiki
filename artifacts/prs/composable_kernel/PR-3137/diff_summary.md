# Diff summary

- **files changed:** 19
- **lines:** +777 / -172
- **kernel-ish files:** 15

## Files (by churn)

- `profiler/include/profiler/common.hpp`  (+103/-0)
- `test/gemm_multiply_multiply_wp/test_gemm_common.hpp`  (+93/-0)
- `profiler/include/profiler/profile_grouped_conv_fwd_outelementop_impl.hpp`  (+1/-82)
- `test/gemm_universal_preshuffle/test_gemm_common.hpp`  (+79/-0)
- `include/ck/tensor_operation/gpu/grid/gridwise_gemm_xdl_cshuffle_v3_b_preshuffle.hpp`  (+43/-34)
- `test/gemm_blockscale_wp/test_gemm_common.hpp`  (+77/-0)
- `test/gemm_multiply_multiply_wp/test_gemm_multiply_multiply_wp_xdl_fp8.cpp`  (+77/-0)
- `test/gemm_universal_preshuffle/test_gemm_universal_preshuffle_xdl_fp8.cpp`  (+77/-0)
- `include/ck/tensor_operation/gpu/grid/gridwise_gemm_xdl_cshuffle_v3_multi_d_b_preshuffle.hpp`  (+42/-25)
- `test/gemm_blockscale_wp/test_gemm_blockscale_wp_xdl_fp8.cpp`  (+64/-0)
- `profiler/include/profiler/profile_gemm_blockscale_wp_impl.hpp`  (+33/-13)
- `profiler/include/profiler/profile_gemm_multiply_multiply_wp_impl.hpp`  (+26/-3)
- `profiler/src/profile_gemm_blockscale_wp.cpp`  (+13/-13)
- `profiler/include/profiler/profile_gemm_universal_preshuffle_impl.hpp`  (+23/-2)
- `test/gemm_blockscale_wp/CMakeLists.txt`  (+6/-0)

## Key added lines (kernel files)

**`include/ck/tensor_operation/gpu/device/impl/device_gemm_xdl_cshuffle_v3_b_preshuffle.hpp`**
```
if(arg.N % NPerBlock != 0 || arg.K % KPerBlock != 0)
return false;
```

**`include/ck/tensor_operation/gpu/grid/gridwise_gemm_xdl_cshuffle_v3_b_preshuffle.hpp`**
```
const index_t Kt = karg.K;
const index_t num_k_per_block = GridwiseGemm::CalculateBK0Shuffled(karg.K);
const index_t k_id            = blockIdx.z * num_k_per_block;
karg.p_b_grid,
```

**`include/ck/tensor_operation/gpu/grid/gridwise_gemm_xdl_cshuffle_v3_multi_d_b_preshuffle.hpp`**
```
const index_t Kt = karg.K;
const index_t num_k_per_block = GridwiseGemm::CalculateBK0Shuffled(karg.K);
const index_t k_id            = blockIdx.z * num_k_per_block;
karg.p_b_grid,
```

**`profiler/include/profiler/common.hpp`**
```
namespace ck {
namespace profiler {
template <typename DataType, typename ComputeDataType = DataType>
inline __host__ __device__ constexpr double get_rtol()
```

**`profiler/include/profiler/profile_gemm_blockscale_wp_impl.hpp`**
```
bool profile_gemm_blockscale_weightpreshuffle_impl(int do_verification,
int init_method,
bool do_log,
bool time_kernel,
```
