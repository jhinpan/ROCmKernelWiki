# Diff summary

- **files changed:** 15 (diff was byte-capped; summary is partial)
- **lines:** +360 / -318
- **kernel-ish files:** 15

## Files (by churn)

- `include/ck/tensor_operation/gpu/device/device_gemm_xdl_cshuffle.hpp`  (+77/-30)
- `library/src/tensor_operation_instance/gpu/gemm/device_gemm_xdl_c_shuffle_bf16_bf16_bf16_km_kn_mn_instance.cpp`  (+27/-26)
- `library/src/tensor_operation_instance/gpu/gemm/device_gemm_xdl_c_shuffle_bf16_bf16_bf16_km_nk_mn_instance.cpp`  (+27/-26)
- `library/src/tensor_operation_instance/gpu/gemm/device_gemm_xdl_c_shuffle_bf16_bf16_bf16_mk_kn_mn_instance.cpp`  (+27/-26)
- `library/src/tensor_operation_instance/gpu/gemm/device_gemm_xdl_c_shuffle_f16_f16_f16_km_kn_mn_instance.cpp`  (+23/-21)
- `library/src/tensor_operation_instance/gpu/gemm/device_gemm_xdl_c_shuffle_f16_f16_f16_km_nk_mn_instance.cpp`  (+23/-21)
- `library/src/tensor_operation_instance/gpu/gemm/device_gemm_xdl_c_shuffle_f16_f16_f16_mk_kn_mn_instance.cpp`  (+23/-21)
- `library/src/tensor_operation_instance/gpu/gemm/device_gemm_xdl_c_shuffle_f32_f32_f32_km_kn_mn_instance.cpp`  (+23/-21)
- `library/src/tensor_operation_instance/gpu/gemm/device_gemm_xdl_c_shuffle_f32_f32_f32_km_nk_mn_instance.cpp`  (+23/-21)
- `library/src/tensor_operation_instance/gpu/gemm/device_gemm_xdl_c_shuffle_f32_f32_f32_mk_kn_mn_instance.cpp`  (+23/-21)
- `library/src/tensor_operation_instance/gpu/gemm/device_gemm_xdl_c_shuffle_bf16_bf16_bf16_mk_nk_mn_instance.cpp`  (+20/-18)
- `library/src/tensor_operation_instance/gpu/gemm/device_gemm_xdl_c_shuffle_f16_f16_f16_mk_nk_mn_instance.cpp`  (+20/-18)
- `library/src/tensor_operation_instance/gpu/gemm/device_gemm_xdl_c_shuffle_f32_f32_f32_mk_nk_mn_instance.cpp`  (+20/-18)
- `example/01_gemm/gemm_xdl_fp16.cpp`  (+1/-16)
- `library/src/tensor_operation_instance/gpu/gemm/device_gemm_xdl_c_shuffle_int8_int8_int8_km_kn_mn_instance.cpp`  (+3/-14)

## Key added lines (kernel files)

**`example/01_gemm/gemm_xdl_fp16.cpp`**
```
float ave_time = invoker.Run(argument, nrepeat);
```

**`include/ck/tensor_operation/gpu/device/device_gemm_xdl_cshuffle.hpp`**
```
float Run(const Argument& arg, int nrepeat = 1)
float ave_time = 0;
if(nrepeat == 0)
launch_kernel(kernel,
```

**`library/src/tensor_operation_instance/gpu/gemm/device_gemm_xdl_c_shuffle_bf16_bf16_bf16_km_kn_mn_instance.cpp`**
```
static constexpr auto GemmDefault = ck::tensor_operation::device::GemmSpecialization_t::Default;
using device_gemm_xdl_c_shuffle_bf16_bf16_bf16_km_kn_mn_instances = std::tuple<
DeviceGemm_Xdl_CShuffle<     Col,      Row,    Row,  BF16,  BF16,  BF16,     F32,     BF16, PassThrough, PassThrough, Pa
DeviceGemm_Xdl_CShuffle<     Col,      Row,    Row,  BF16,  BF16,  BF16,     F32,     BF16, PassThrough, PassThrough, Pa
```

**`library/src/tensor_operation_instance/gpu/gemm/device_gemm_xdl_c_shuffle_bf16_bf16_bf16_km_nk_mn_instance.cpp`**
```
static constexpr auto GemmDefault = ck::tensor_operation::device::GemmSpecialization_t::Default;
using device_gemm_xdl_c_shuffle_bf16_bf16_bf16_km_nk_mn_instances = std::tuple<
DeviceGemm_Xdl_CShuffle<     Col,      Col,    Row,  BF16,  BF16,  BF16,     F32,     BF16, PassThrough, PassThrough, Pa
DeviceGemm_Xdl_CShuffle<     Col,      Col,    Row,  BF16,  BF16,  BF16,     F32,     BF16, PassThrough, PassThrough, Pa
```

**`library/src/tensor_operation_instance/gpu/gemm/device_gemm_xdl_c_shuffle_bf16_bf16_bf16_mk_kn_mn_instance.cpp`**
```
static constexpr auto GemmDefault = ck::tensor_operation::device::GemmSpecialization_t::Default;
using device_gemm_xdl_c_shuffle_bf16_bf16_bf16_mk_kn_mn_instances = std::tuple<
DeviceGemm_Xdl_CShuffle<     Row,      Row,    Row,  BF16,  BF16,  BF16,     F32,     BF16, PassThrough, PassThrough, Pa
DeviceGemm_Xdl_CShuffle<     Row,      Row,    Row,  BF16,  BF16,  BF16,     F32,     BF16, PassThrough, PassThrough, Pa
```
