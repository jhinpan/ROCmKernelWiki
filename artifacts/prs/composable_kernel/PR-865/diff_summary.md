# Diff summary

- **files changed:** 22
- **lines:** +2082 / -27
- **kernel-ish files:** 20

## Files (by churn)

- `include/ck/tensor_operation/gpu/grid/gridwise_gemm_split_k_multiple_d_xdl_cshuffle_v2.hpp`  (+1076/-0)
- `profiler/include/profiler/profile_gemm_multiply_add_impl.hpp`  (+242/-0)
- `library/include/ck/library/tensor_operation_instance/gpu/gemm_multiply_add.hpp`  (+161/-0)
- `profiler/src/profile_gemm_multiply_add.cpp`  (+153/-0)
- `library/src/tensor_operation_instance/gpu/gemm_multiply_add/device_gemm_multiply_add_xdl_c_shuffle_f16_f8_f32_f32_f16_mk_kn_mn_mn_mn_instance.cpp`  (+84/-0)
- `library/src/tensor_operation_instance/gpu/gemm_multiply_add/device_gemm_multiply_add_xdl_c_shuffle_f16_f16_f16_f16_f16_mk_kn_mn_mn_mn_instance.cpp`  (+83/-0)
- `library/src/tensor_operation_instance/gpu/gemm_multiply_add/device_gemm_multiply_add_xdl_c_shuffle_f16_f8_f32_f32_f16_mk_nk_mn_mn_mn_instance.cpp`  (+83/-0)
- `library/src/tensor_operation_instance/gpu/gemm_multiply_add/device_gemm_multiply_add_xdl_c_shuffle_f16_f16_f16_f16_f16_mk_nk_mn_mn_mn_instance.cpp`  (+82/-0)
- `include/ck/tensor_operation/gpu/element/element_wise_operation.hpp`  (+45/-0)
- `include/ck/tensor_operation/gpu/grid/gridwise_gemm_multiple_d_xdl_cshuffle.hpp`  (+19/-22)
- `include/ck/tensor_operation/gpu/device/impl/device_gemm_multiple_d_xdl_cshuffle.hpp`  (+9/-3)
- `include/ck/tensor_operation/gpu/device/impl/device_batched_gemm_e_permute_xdl.hpp`  (+6/-1)
- `library/src/tensor_operation_instance/gpu/gemm_multiply_add/CMakeLists.txt`  (+7/-0)
- `include/ck/tensor_operation/gpu/device/impl/device_batched_contraction_multiple_d_xdl_cshuffle.hpp`  (+5/-1)
- `include/ck/tensor_operation/gpu/device/impl/device_batched_gemm_multi_d_xdl.hpp`  (+4/-0)

## Key added lines (kernel files)

**`include/ck/tensor_operation/gpu/device/impl/device_batched_contraction_multiple_d_xdl_cshuffle.hpp`**
```
using ComputeDataType = ADataType;
ADataType,
BDataType,
ComputeDataType,
```

**`include/ck/tensor_operation/gpu/device/impl/device_batched_gemm_e_permute_xdl.hpp`**
```
using ComputeDataType = ADataType;
ADataType,
BDataType,
ComputeDataType,
```

**`include/ck/tensor_operation/gpu/device/impl/device_batched_gemm_multi_d_xdl.hpp`**
```
using ComputeDataType = ADataType;
BDataType,
ComputeDataType,
```

**`include/ck/tensor_operation/gpu/device/impl/device_contraction_multiple_d_xdl_cshuffle.hpp`**
```
using ComputeDataType = ADataType;
BDataType,
ComputeDataType,
```

**`include/ck/tensor_operation/gpu/device/impl/device_gemm_multiple_d_xdl_cshuffle.hpp`**
```
typename ADataType,
typename BDataType,
kernel_gemm_multiple_d_xdl_cshuffle(const ADataType* __restrict__ p_a_grid,
const BDataType* __restrict__ p_b_grid,
```
