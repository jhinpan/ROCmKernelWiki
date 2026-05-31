# Diff summary

- **files changed:** 14
- **lines:** +191 / -151
- **kernel-ish files:** 14

## Files (by churn)

- `device_operation/src/device_gemm_xdl_c_shuffle_f16_f16_f16_km_kn_mn_instance.cpp`  (+20/-20)
- `device_operation/src/device_gemm_xdl_c_shuffle_f16_f16_f16_km_nk_mn_instance.cpp`  (+20/-20)
- `device_operation/src/device_gemm_xdl_c_shuffle_f16_f16_f16_mk_kn_mn_instance.cpp`  (+20/-20)
- `device_operation/src/device_gemm_xdl_c_shuffle_2_stage_f16_f16_f16_mk_nk_mn_instance.cpp`  (+17/-17)
- `device_operation/src/device_gemm_xdl_c_shuffle_bf16_bf16_bf16_mk_nk_mn_instance.cpp`  (+17/-17)
- `device_operation/src/device_gemm_xdl_c_shuffle_f16_f16_f16_mk_nk_mn_instance.cpp`  (+17/-17)
- `device_operation/src/device_gemm_xdl_c_shuffle_int8_int8_int8_mk_nk_mn_instance.cpp`  (+17/-17)
- `composable_kernel/include/tensor_operation/element_wise_operation.hpp`  (+31/-0)
- `example/1_gemm_xdl/gemm_xdl_int8.cpp`  (+14/-8)
- `composable_kernel/include/tensor_operation/gridwise_gemm_xdlops_v3r1.hpp`  (+9/-10)
- `example/1_gemm_xdl/gemm_xdl.cpp`  (+5/-5)
- `device_operation/include/device_gemm_xdl_c_shuffle.hpp`  (+2/-0)
- `device_operation/include/device_conv2d_fwd_xdl_c_shuffle_nhwc_kyxc_nhwk.hpp`  (+1/-0)
- `example/1_gemm_xdl/gemm_xdl_bf16.cpp`  (+1/-0)

## Key added lines (kernel files)

**`composable_kernel/include/tensor_operation/element_wise_operation.hpp`**
```
struct RequantReluRequant
RequantReluRequant(float scaleGemm, float scaleRelu)
: scaleGemm_(scaleGemm), scaleRelu_(scaleRelu)
__host__ __device__ constexpr void operator()(int8_t& y, const int& x) const
```

**`composable_kernel/include/tensor_operation/gridwise_gemm_xdlops_v3r1.hpp`**
```
typename FloatCShuffle,
c_block_size * sizeof(FloatCShuffle));
auto c_shuffle_block_buf = make_dynamic_buffer<AddressSpaceEnum_t::Lds>(
static_cast<FloatCShuffle*>(p_shared),
```

**`device_operation/include/device_conv2d_fwd_xdl_c_shuffle_nhwc_kyxc_nhwk.hpp`**
```
CDataType, // TODO: Add ShuffleType for DeviceConv2d
```

**`device_operation/include/device_gemm_xdl_c_shuffle.hpp`**
```
typename CShuffleDataType,
CShuffleDataType,
```

**`device_operation/src/device_gemm_xdl_c_shuffle_2_stage_f16_f16_f16_mk_nk_mn_instance.cpp`**
```
DeviceGemmXdl_C_Shuffle<   F16,   F16,   F16,     F32,      F16,     Row,     Col,     Row, PassThrough, PassThrough, Pa
DeviceGemmXdl_C_Shuffle<   F16,   F16,   F16,     F32,      F16,     Row,     Col,     Row, PassThrough, PassThrough, Pa
DeviceGemmXdl_C_Shuffle<   F16,   F16,   F16,     F32,      F16,     Row,     Col,     Row, PassThrough, PassThrough, Pa
DeviceGemmXdl_C_Shuffle<   F16,   F16,   F16,     F32,      F16,     Row,     Col,     Row, PassThrough, PassThrough, Pa
```
