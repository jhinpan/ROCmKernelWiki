# Diff summary

- **files changed:** 16
- **lines:** +16 / -16
- **kernel-ish files:** 16

## Files (by churn)

- `library/src/tensor_operation_instance/gpu/gemm/device_gemm_xdl_c_shuffle_bf16_bf16_bf16_km_kn_mn_instance.cpp`  (+1/-1)
- `library/src/tensor_operation_instance/gpu/gemm/device_gemm_xdl_c_shuffle_bf16_bf16_bf16_km_nk_mn_instance.cpp`  (+1/-1)
- `library/src/tensor_operation_instance/gpu/gemm/device_gemm_xdl_c_shuffle_bf16_bf16_bf16_mk_kn_mn_instance.cpp`  (+1/-1)
- `library/src/tensor_operation_instance/gpu/gemm/device_gemm_xdl_c_shuffle_bf16_bf16_bf16_mk_nk_mn_instance.cpp`  (+1/-1)
- `library/src/tensor_operation_instance/gpu/gemm/device_gemm_xdl_c_shuffle_f16_f16_f16_km_kn_mn_instance.cpp`  (+1/-1)
- `library/src/tensor_operation_instance/gpu/gemm/device_gemm_xdl_c_shuffle_f16_f16_f16_km_nk_mn_instance.cpp`  (+1/-1)
- `library/src/tensor_operation_instance/gpu/gemm/device_gemm_xdl_c_shuffle_f16_f16_f16_mk_kn_mn_instance.cpp`  (+1/-1)
- `library/src/tensor_operation_instance/gpu/gemm/device_gemm_xdl_c_shuffle_f16_f16_f16_mk_nk_mn_instance.cpp`  (+1/-1)
- `library/src/tensor_operation_instance/gpu/gemm/device_gemm_xdl_c_shuffle_f32_f32_f32_km_kn_mn_instance.cpp`  (+1/-1)
- `library/src/tensor_operation_instance/gpu/gemm/device_gemm_xdl_c_shuffle_f32_f32_f32_km_nk_mn_instance.cpp`  (+1/-1)
- `library/src/tensor_operation_instance/gpu/gemm/device_gemm_xdl_c_shuffle_f32_f32_f32_mk_kn_mn_instance.cpp`  (+1/-1)
- `library/src/tensor_operation_instance/gpu/gemm/device_gemm_xdl_c_shuffle_f32_f32_f32_mk_nk_mn_instance.cpp`  (+1/-1)
- `library/src/tensor_operation_instance/gpu/gemm/device_gemm_xdl_c_shuffle_int8_int8_int8_km_kn_mn_instance.cpp`  (+1/-1)
- `library/src/tensor_operation_instance/gpu/gemm/device_gemm_xdl_c_shuffle_int8_int8_int8_km_nk_mn_instance.cpp`  (+1/-1)
- `library/src/tensor_operation_instance/gpu/gemm/device_gemm_xdl_c_shuffle_int8_int8_int8_mk_kn_mn_instance.cpp`  (+1/-1)

## Key added lines (kernel files)

**`library/src/tensor_operation_instance/gpu/gemm/device_gemm_xdl_c_shuffle_bf16_bf16_bf16_km_kn_mn_instance.cpp`**
```
static constexpr auto GemmDefault = ck::tensor_operation::device::GemmSpecialization::Default;
```

**`library/src/tensor_operation_instance/gpu/gemm/device_gemm_xdl_c_shuffle_bf16_bf16_bf16_km_nk_mn_instance.cpp`**
```
static constexpr auto GemmDefault = ck::tensor_operation::device::GemmSpecialization::Default;
```

**`library/src/tensor_operation_instance/gpu/gemm/device_gemm_xdl_c_shuffle_bf16_bf16_bf16_mk_kn_mn_instance.cpp`**
```
static constexpr auto GemmDefault = ck::tensor_operation::device::GemmSpecialization::Default;
```

**`library/src/tensor_operation_instance/gpu/gemm/device_gemm_xdl_c_shuffle_bf16_bf16_bf16_mk_nk_mn_instance.cpp`**
```
static constexpr auto GemmDefault = ck::tensor_operation::device::GemmSpecialization::Default;
```

**`library/src/tensor_operation_instance/gpu/gemm/device_gemm_xdl_c_shuffle_f16_f16_f16_km_kn_mn_instance.cpp`**
```
static constexpr auto GemmDefault = ck::tensor_operation::device::GemmSpecialization::Default;
```
