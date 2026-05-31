# Diff summary

- **files changed:** 24
- **lines:** +1482 / -307
- **kernel-ish files:** 22

## Files (by churn)

- `test/gemm/gemm_util.hpp`  (+241/-0)
- `test/gemm/gemm_fp32.cpp`  (+102/-86)
- `test/gemm/gemm_bf16.cpp`  (+57/-108)
- `test/gemm/gemm_int8.cpp`  (+75/-84)
- `test/gemm/gemm_fp16.cpp`  (+154/-0)
- `profiler/src/profile_gemm.cpp`  (+120/-0)
- `profiler/include/profile_gemm_impl.hpp`  (+73/-7)
- `library/src/tensor_operation_instance/gpu/gemm/device_gemm_xdl_c_shuffle_bf16_bf16_bf16_km_kn_mn_instance.cpp`  (+59/-0)
- `library/src/tensor_operation_instance/gpu/gemm/device_gemm_xdl_c_shuffle_bf16_bf16_bf16_km_nk_mn_instance.cpp`  (+59/-0)
- `library/src/tensor_operation_instance/gpu/gemm/device_gemm_xdl_c_shuffle_bf16_bf16_bf16_mk_kn_mn_instance.cpp`  (+59/-0)
- `library/src/tensor_operation_instance/gpu/gemm/device_gemm_xdl_c_shuffle_f32_f32_f32_km_kn_mn_instance.cpp`  (+58/-0)
- `library/src/tensor_operation_instance/gpu/gemm/device_gemm_xdl_c_shuffle_f32_f32_f32_km_nk_mn_instance.cpp`  (+58/-0)
- `library/src/tensor_operation_instance/gpu/gemm/device_gemm_xdl_c_shuffle_f32_f32_f32_mk_kn_mn_instance.cpp`  (+58/-0)
- `library/src/tensor_operation_instance/gpu/gemm/device_gemm_xdl_c_shuffle_int8_int8_int8_km_kn_mn_instance.cpp`  (+58/-0)
- `library/src/tensor_operation_instance/gpu/gemm/device_gemm_xdl_c_shuffle_int8_int8_int8_km_nk_mn_instance.cpp`  (+58/-0)

## Key added lines (kernel files)

**`example/02_gemm_alpha_beta/gemm_xdl_alpha_beta.cpp`**
```
Tensor<CDataType> c0_m_n(f_host_tensor_descriptor(M, N, StrideC, CLayout{}));
Tensor<CDataType> c_m_n_host_result(f_host_tensor_descriptor(M, N, StrideC, CLayout{}));
Tensor<CDataType> c_m_n_device_result(f_host_tensor_descriptor(M, N, StrideC, CLayout{}));
c0_m_n.GenerateTensorValue(GeneratorTensor_2<CDataType>{-5, 5});
```

**`example/03_gemm_bias_relu/gemm_xdl_bias_relu.cpp`**
```
Tensor<CDataType> c_m_n_host_result(f_host_tensor_descriptor(M, N, StrideC, CLayout{}));
Tensor<CDataType> c_m_n_device_result(f_host_tensor_descriptor(M, N, StrideC, CLayout{}));
```

**`example/04_gemm_bias_relu_add/gemm_xdl_bias_relu_add.cpp`**
```
Tensor<CDataType> c_m_n_host_result(f_host_tensor_descriptor(M, N, StrideC, CLayout{}));
Tensor<CDataType> c_m_n_device_result(f_host_tensor_descriptor(M, N, StrideC, CLayout{}));
Tensor<CDataType> c1_m_n(f_host_tensor_descriptor(M, N, StrideC, CLayout{}));
```

**`library/src/tensor_operation_instance/gpu/gemm/device_gemm_xdl_c_shuffle_bf16_bf16_bf16_km_kn_mn_instance.cpp`**
```
namespace ck {
namespace tensor_operation {
namespace device {
namespace device_gemm_instance {
```

**`library/src/tensor_operation_instance/gpu/gemm/device_gemm_xdl_c_shuffle_bf16_bf16_bf16_km_nk_mn_instance.cpp`**
```
namespace ck {
namespace tensor_operation {
namespace device {
namespace device_gemm_instance {
```
