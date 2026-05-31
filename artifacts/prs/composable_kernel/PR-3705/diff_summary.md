# Diff summary

- **files changed:** 24
- **lines:** +120 / -3517
- **kernel-ish files:** 21

## Files (by churn)

- `include/ck/tensor_operation/gpu/device/impl/device_grouped_gemm_multi_abd_wmma_fixed_nk.hpp`  (+0/-899)
- `profiler/include/profiler/profile_grouped_gemm_multi_abd_fixed_nk_impl.hpp`  (+0/-534)
- `example/59_grouped_gemm_multi_ABD/grouped_gemm_multi_abd_wmma_fixed_nk_bias_bf16_i8.cpp`  (+0/-400)
- `example/59_grouped_gemm_multi_ABD/grouped_gemm_multi_abd_wmma_fixed_nk_bias_fp16.cpp`  (+0/-396)
- `library/include/ck/library/tensor_operation_instance/gpu/grouped_gemm_multi_abd_fixed_nk.hpp`  (+1/-294)
- `test/grouped_gemm/test_grouped_gemm_multi_abd_fixed_nk.cpp`  (+0/-256)
- `library/include/ck/library/reference_tensor_operation/cpu/reference_gemm_multi_abd.hpp`  (+0/-194)
- `library/src/tensor_operation_instance/gpu/grouped_gemm_fixed_nk_multi_abd/device_grouped_gemm_wmma_fixed_nk_bias_gelu_bf16_i8_bf16_km_kn_mn_instance.cpp`  (+0/-144)
- `library/src/tensor_operation_instance/gpu/grouped_gemm_fixed_nk_multi_abd/device_grouped_gemm_wmma_fixed_nk_bias_gelu_bf16_i8_bf16_mk_kn_mn_instance.cpp`  (+0/-144)
- `library/src/tensor_operation_instance/gpu/grouped_gemm_fixed_nk_multi_abd/device_grouped_gemm_wmma_fixed_nk_bias_gelu_bf16_i8_bf16_mk_nk_mn_instance.cpp`  (+0/-144)
- `profiler/include/profiler/profile_gemm_multi_abd_impl.hpp`  (+72/-16)
- `example/59_grouped_gemm_multi_ABD/grouped_gemm_multi_abd_xdl_fixed_nk_bias_fp16.cpp`  (+19/-14)
- `example/59_grouped_gemm_multi_ABD/grouped_gemm_multi_abd_xdl_fixed_nk_bias_bf16_i8.cpp`  (+17/-10)
- `include/ck/tensor_operation/gpu/device/impl/device_grouped_gemm_multi_abd_xdl_fixed_nk.hpp`  (+1/-26)
- `library/src/tensor_operation_instance/gpu/grouped_gemm_fixed_nk_multi_abd/device_grouped_gemm_xdl_fixed_nk_bf16_i8_bf16_km_kn_mn_common.hpp`  (+4/-6)

## Key added lines (kernel files)

**`example/59_grouped_gemm_multi_ABD/grouped_gemm_multi_abd_xdl_fixed_nk_bias_bf16_i8.cpp`**
```
a0_tensors_device.emplace_back(
std::make_unique<DeviceMem>(sizeof(A0DataType) * sum_of_m * problem_size.Ks[i]));
c_tensors_device.emplace_back(
std::make_unique<DeviceMem>(sizeof(EDataType) * sum_of_m * problem_size.Ns[i]));
```

**`example/59_grouped_gemm_multi_ABD/grouped_gemm_multi_abd_xdl_fixed_nk_bias_fp16.cpp`**
```
using D0DataType       = F32;
using EDataType        = F32;
a0_tensors_device.emplace_back(
std::make_unique<DeviceMem>(sizeof(A0DataType) * sum_of_m * problem_size.Ks[i]));
```

**`include/ck/tensor_operation/gpu/device/impl/device_grouped_gemm_multi_abd_xdl_fixed_nk.hpp`**
```
throw std::runtime_error("wrong! grouped_gemm_kernel_args_dev is nullpr");
```

**`include/ck/tensor_operation/gpu/device/impl/device_grouped_gemm_xdl_fixed_nk.hpp`**
```
throw std::runtime_error("wrong! grouped_gemm_kernel_args_dev is nullpr");
```

**`library/src/tensor_operation_instance/gpu/grouped_gemm_fixed_nk_multi_abd/device_grouped_gemm_xdl_fixed_nk_bf16_i8_bf16_km_kn_mn_common.hpp`**
```
DeviceGroupedGemm_Xdl_Multi_ABD_Fixed_NK< AsLayout, BsLayout, DsLayout, ELayout, AsDataType, BsDataType, AccDataType, CS
DeviceGroupedGemm_Xdl_Multi_ABD_Fixed_NK< AsLayout, BsLayout, DsLayout, ELayout, AsDataType, BsDataType, AccDataType, CS
DeviceGroupedGemm_Xdl_Multi_ABD_Fixed_NK< AsLayout, BsLayout, DsLayout, ELayout, AsDataType, BsDataType, AccDataType, CS
DeviceGroupedGemm_Xdl_Multi_ABD_Fixed_NK< AsLayout, BsLayout, DsLayout, ELayout, AsDataType, BsDataType, AccDataType, CS
```
