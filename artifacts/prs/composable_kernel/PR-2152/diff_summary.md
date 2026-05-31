# Diff summary

- **files changed:** 15
- **lines:** +642 / -18
- **kernel-ish files:** 12

## Files (by churn)

- `test/gemm_mx/test_gemm_mx.cpp`  (+174/-5)
- `include/ck/utility/amd_xdlops.hpp`  (+98/-3)
- `example/67_gemm_microscaling/gemm_mx_fp8_bf8.cpp`  (+97/-0)
- `library/src/tensor_operation_instance/gpu/gemm_mx/device_gemm_mx_xdl_f8_f8_bf16/device_gemm_mx_xdl_f8_f8_bf16_km_nk_mn.hpp`  (+62/-0)
- `library/src/tensor_operation_instance/gpu/gemm_mx/device_gemm_mx_xdl_bf8_f8_f16/device_gemm_mx_xdl_bf8_f8_f16_mk_kn_mn.hpp`  (+61/-0)
- `library/include/ck/library/tensor_operation_instance/gpu/gemm_mx.hpp`  (+49/-1)
- `library/src/tensor_operation_instance/gpu/gemm_mx/device_gemm_mx_xdl_bf8_f8_f16/device_gemm_mx_xdl_bf8_f8_f16_mk_kn_mn_default_instance.cpp`  (+32/-0)
- `library/src/tensor_operation_instance/gpu/gemm_mx/device_gemm_mx_xdl_f8_f8_bf16/device_gemm_mx_xdl_f8_f8_bf16_km_nk_mn_default_instance.cpp`  (+32/-0)
- `include/ck/tensor_operation/gpu/warp/xdlops_gemm.hpp`  (+18/-0)
- `include/ck/tensor_operation/gpu/grid/gridwise_gemm_xdl_cshuffle_v3_mx.hpp`  (+8/-6)
- `library/src/tensor_operation_instance/gpu/gemm_mx/CMakeLists.txt`  (+5/-1)
- `example/67_gemm_microscaling/CMakeLists.txt`  (+3/-0)
- `example/67_gemm_microscaling/gemm_mx_common.hpp`  (+1/-1)
- `test/gemm_mx/test_gemm_mx_util.hpp`  (+1/-1)
- `test/CMakeLists.txt`  (+1/-0)

## Key added lines (kernel files)

**`example/67_gemm_microscaling/gemm_mx_common.hpp`**
```
a_m_k.GenerateTensorValue(GeneratorTensor_3<ADataType>{-2.0, 2.0});
```

**`example/67_gemm_microscaling/gemm_mx_fp8_bf8.cpp`**
```
using ADataType = ck::f8_t;
using BDataType = ck::bf8_t;
using XDataType = ck::e8m0_bexp_t;
using CDataType        = ck::bhalf_t;
```

**`include/ck/tensor_operation/gpu/grid/gridwise_gemm_xdl_cshuffle_v3_mx.hpp`**
```
constexpr auto WaveSize = 64;
constexpr auto M0       = ABlockTransferThreadClusterLengths_AK0_M_AK1{}.At(I1);
constexpr auto M1       = MPerBlock / M0;
constexpr auto KThreadRead      = WaveSize / MPerXdl;
```

**`include/ck/tensor_operation/gpu/warp/xdlops_gemm.hpp`**
```
template <>
constexpr auto GetMfma<bf8_t, 32, 32, f8_t, false, true>()
return MfmaInstr::mfma_scale_f32_32x32x64f8f6f4;
template <>
```

**`include/ck/utility/amd_xdlops.hpp`**
```
0, // cbsz {0 FP8 E4M3; 1 FP8 E5M2; 2 FP6 E2M3; 3 FP6 E3M2; 4 FP4 E2M1}
0, // blgp
0, // OPSEL
0, // OPSEL
```
