# Diff summary

- **files changed:** 66 (diff was byte-capped; summary is partial)
- **lines:** +1175 / -1254
- **kernel-ish files:** 51

## Files (by churn)

- `include/ck/tensor_operation/gpu/device/impl/device_gemm_multiple_d_xdl_cshuffle_v3.hpp`  (+282/-99)
- `include/ck/tensor_operation/gpu/device/impl/device_gemm_xdl_cshuffle_v3.hpp`  (+142/-162)
- `library/include/ck/library/tensor_operation_instance/gpu/gemm_multiply_multiply.hpp`  (+79/-138)
- `library/include/ck/library/tensor_operation_instance/gpu/gemm_universal.hpp`  (+56/-129)
- `library/src/tensor_operation_instance/gpu/gemm_universal/CMakeLists.txt`  (+88/-61)
- `library/src/tensor_operation_instance/gpu/gemm_universal/device_gemm_xdl_universal_f8_f8_bf16/device_gemm_xdl_universal_f8_f8_bf16_mk_kn_mn.hpp`  (+96/-0)
- `include/ck/utility/amd_buffer_addressing.hpp`  (+47/-8)
- `include/ck/tensor_operation/gpu/device/device_gemm_multiple_d.hpp`  (+43/-0)
- `library/src/tensor_operation_instance/gpu/gemm_multiply_multiply/device_gemm_multiply_multiply_xdl_f8_f8_bf16/device_gemm_multiply_multiply_xdl_f8_f8_bf16_mk_nk_mn_mem_v1_mnkpadding_instance.cpp`  (+0/-33)
- `library/src/tensor_operation_instance/gpu/gemm_multiply_multiply/device_gemm_multiply_multiply_xdl_f8_f8_bf16/device_gemm_multiply_multiply_xdl_f8_f8_bf16_mk_nk_mn_mem_v2_mnkpadding_instance.cpp`  (+0/-33)
- `library/src/tensor_operation_instance/gpu/gemm_multiply_multiply/device_gemm_multiply_multiply_xdl_f8_f8_bf16/device_gemm_multiply_multiply_xdl_f8_f8_bf16_mk_nk_mn_comp_mnkpadding_instance.cpp`  (+0/-32)
- `library/src/tensor_operation_instance/gpu/gemm_multiply_multiply/device_gemm_multiply_multiply_xdl_f8_f8_bf16/device_gemm_multiply_multiply_xdl_f8_f8_bf16_mk_nk_mn_comp_mnpadding_instance.cpp`  (+0/-32)
- `include/ck/tensor_operation/gpu/grid/gridwise_gemm_xdl_cshuffle_v3_multi_d.hpp`  (+18/-10)
- `library/src/tensor_operation_instance/gpu/gemm_universal/device_gemm_xdl_universal_f16_f8_f16/device_gemm_xdl_universal_f16_f8_f16_mk_nk_mn_comp_mnpadding_instance.cpp`  (+0/-26)
- `library/src/tensor_operation_instance/gpu/gemm_universal/device_gemm_xdl_universal_f8_f16_f16/device_gemm_xdl_universal_f8_f16_f16_mk_nk_mn_comp_mnpadding_instance.cpp`  (+0/-26)

## Key added lines (kernel files)

**`example/20_grouped_conv_bwd_weight/common.hpp`**
```
using F8   = ck::f8_t;
using BF8  = ck::bf8_t;
```

**`example/65_gemm_multiply_multiply/gemm_multiply_multiply_xdl_fp8.cpp`**
```
static constexpr auto GemmSpec = ck::tensor_operation::device::GemmSpecialization::MNPadding;
ck::index_t KBatch = 1;
else if(argc == 12)
KBatch = std::stoi(argv[11]);
```

**`include/ck/host_utility/device_prop.hpp`**
```
inline bool is_bf16_atomic_supported()
return ck::get_device_name() == "gfx940" || ck::get_device_name() == "gfx941" ||
ck::get_device_name() == "gfx942";
```

**`include/ck/tensor_operation/gpu/device/device_gemm_multiple_d.hpp`**
```
template <typename ALayout,
typename BLayout,
typename DsLayout,
typename ELayout,
```

**`include/ck/tensor_operation/gpu/device/impl/device_gemm_multiple_d_xdl_cshuffle_v3.hpp`**
```
struct DeviceGemmMultiD_Xdl_CShuffle_V3 : public DeviceGemmMultipleDSplitK<ALayout,
DsLayout,
ADataType,
BDataType,
```
