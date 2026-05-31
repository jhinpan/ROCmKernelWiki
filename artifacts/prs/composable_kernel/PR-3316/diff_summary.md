# Diff summary

- **files changed:** 15
- **lines:** +1426 / -143
- **kernel-ish files:** 12

## Files (by churn)

- `include/ck/tensor_operation/gpu/device/impl/device_gemm_bias_add_reduce_wmma_cshuffle_v3.hpp`  (+682/-0)
- `include/ck/tensor_operation/gpu/grid/epilogue_cshuffle_v3_reduce_wmma.hpp`  (+128/-38)
- `profiler/include/profiler/profile_gemm_bias_add_reduce_impl.hpp`  (+50/-102)
- `test/gemm_bias_add_reduce/test_gemm_bias_add_reduce_fp16.cpp`  (+106/-0)
- `library/src/tensor_operation_instance/gpu/gemm_bias_add_reduce/device_gemm_bias_add_mean_squaremean_wmma_cshuffle_f16_f16_f16_f32_f32_km_kn_mn_instance.cpp`  (+85/-0)
- `library/src/tensor_operation_instance/gpu/gemm_bias_add_reduce/device_gemm_bias_add_mean_squaremean_wmma_cshuffle_f16_f16_f16_f32_f32_km_nk_mn_instance.cpp`  (+84/-0)
- `library/src/tensor_operation_instance/gpu/gemm_bias_add_reduce/device_gemm_bias_add_mean_squaremean_wmma_cshuffle_f16_f16_f16_f32_f32_mk_kn_mn_instance.cpp`  (+84/-0)
- `library/src/tensor_operation_instance/gpu/gemm_bias_add_reduce/device_gemm_bias_add_mean_squaremean_wmma_cshuffle_f16_f16_f16_f32_f32_mk_nk_mn_instance.cpp`  (+81/-0)
- `test/gemm_bias_add_reduce/test_gemm_common.hpp`  (+61/-0)
- `library/include/ck/library/tensor_operation_instance/gpu/device_gemm_mean_squaremean_instance.hpp`  (+41/-0)
- `test/gemm_bias_add_reduce/CMakeLists.txt`  (+9/-0)
- `include/ck/tensor_operation/gpu/device/impl/device_gemm_reduce_wmma_cshuffle_v3.hpp`  (+6/-2)
- `library/src/tensor_operation_instance/gpu/gemm_bias_add_reduce/CMakeLists.txt`  (+6/-1)
- `include/ck/tensor_operation/gpu/grid/gridwise_gemm_bias_add_reduce_xdl_cshuffle_v1.hpp`  (+2/-0)
- `test/CMakeLists.txt`  (+1/-0)

## Key added lines (kernel files)

**`include/ck/tensor_operation/gpu/device/impl/device_gemm_bias_add_reduce_wmma_cshuffle_v3.hpp`**
```
namespace ck {
template <typename GridwiseGemm,
typename ReduceTrait,
bool HasMainKBlockLoop,
```

**`include/ck/tensor_operation/gpu/device/impl/device_gemm_reduce_wmma_cshuffle_v3.hpp`**
```
auto epilogue_args = EpilogueType(p_reduces_grid,
reduce_in_element_ops,
reduce_out_element_ops,
tensor_operation::element_wise::PassThrough{});
```

**`include/ck/tensor_operation/gpu/grid/epilogue_cshuffle_v3_reduce_wmma.hpp`**
```
typename D0ElementwiseOperation,
using D0ElementwiseOperation_          = D0ElementwiseOperation;
const index_t MRaw_,
const typename ReduceTrait::D0ElementwiseOperation_ d0_element_op_)
```

**`include/ck/tensor_operation/gpu/grid/gridwise_gemm_bias_add_reduce_xdl_cshuffle_v1.hpp`**
```
block_sync_lds();
```

**`library/include/ck/library/tensor_operation_instance/gpu/device_gemm_mean_squaremean_instance.hpp`**
```
void add_device_gemm_bias_add_mean_squaremean_wmma_cshuffle_f16_f16_f16_f16_f16_f32_f32_mk_kn_mn_instances(
std::vector<DeviceGemmAddAddMeanSquareMeanPtr>&);
void add_device_gemm_bias_add_mean_squaremean_wmma_cshuffle_f16_f16_f16_f16_f16_f32_f32_mk_nk_mn_instances(
std::vector<DeviceGemmAddAddMeanSquareMeanPtr>&);
```
