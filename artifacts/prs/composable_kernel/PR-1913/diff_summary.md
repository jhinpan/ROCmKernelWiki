# Diff summary

- **files changed:** 18
- **lines:** +1017 / -662
- **kernel-ish files:** 16

## Files (by churn)

- `include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_v1_ab_scale.hpp`  (+532/-83)
- `include/ck/tensor_operation/gpu/grid/gridwise_gemm_xdl_cshuffle_v3_multi_d_ab_scale.hpp`  (+105/-123)
- `include/ck/tensor_operation/gpu/device/impl/device_gemm_multiple_d_xdl_cshuffle_v3_ab_scale.hpp`  (+69/-128)
- `include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_v3_ab_scale.hpp`  (+120/-33)
- `include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_v2_ab_scale.hpp`  (+68/-25)
- `library/include/ck/library/tensor_operation_instance/gpu/gemm_ab_scale.hpp`  (+14/-74)
- `example/65_gemm_multiply_multiply/gemm_multiply_multiply_xdl_fp8_ab_scale.cpp`  (+52/-20)
- `library/src/tensor_operation_instance/gpu/gemm_ab_scale/device_gemm_ab_scale_xdl_f8_f8_bf16/device_gemm_ab_scale_xdl_f8_f8_bf16_mk_nk_mn_128_128_128.hpp`  (+36/-35)
- `library/src/tensor_operation_instance/gpu/gemm_ab_scale/device_gemm_ab_scale_xdl_f8_f8_bf16/device_gemm_ab_scale_xdl_f8_f8_bf16_mk_nk_mn_128_128_128_mem_v1_mnkpadding_instance.cpp`  (+0/-38)
- `library/src/tensor_operation_instance/gpu/gemm_ab_scale/device_gemm_ab_scale_xdl_f8_f8_bf16/device_gemm_ab_scale_xdl_f8_f8_bf16_mk_nk_mn_128_128_128_comp_mnkpadding_instance.cpp`  (+0/-37)
- `library/src/tensor_operation_instance/gpu/gemm_ab_scale/device_gemm_ab_scale_xdl_f8_f8_bf16/device_gemm_ab_scale_xdl_f8_f8_bf16_mk_nk_mn_128_128_128_comp_mnpadding_instance.cpp`  (+0/-37)
- `library/src/tensor_operation_instance/gpu/gemm_ab_scale/device_gemm_ab_scale_xdl_f8_f8_bf16/device_gemm_ab_scale_xdl_f8_f8_bf16_mk_nk_mn_128_128_128_mem_v1_default_instance.cpp`  (+4/-4)
- `library/src/tensor_operation_instance/gpu/gemm_ab_scale/device_gemm_ab_scale_xdl_f8_f8_bf16/device_gemm_ab_scale_xdl_f8_f8_bf16_mk_nk_mn_128_128_128_mem_v1_kpadding_instance.cpp`  (+4/-4)
- `profiler/src/profile_gemm_ab_scale.cpp`  (+5/-3)
- `CMakeLists.txt`  (+0/-7)

## Key added lines (kernel files)

**`example/65_gemm_multiply_multiply/gemm_multiply_multiply_xdl_fp8_ab_scale.cpp`**
```
static constexpr ck::index_t Scale_Block_M = 1;
256, 16, 16,
S<16, 16, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 16, 16, 0,
S<16, 16, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 16, 16, 0,
```

**`include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_v1_ab_scale.hpp`**
```
using Base::A_K1;
using Base::B_K1;
using Base::I1;
using typename Base::HotLoopInstList;
```

**`include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_v2_ab_scale.hpp`**
```
static_for<0, MRepeat, 1>{}([&](auto m0) {
a_scale_thread_copy.Run(a_scale_grid_desc,
a_scale_grid_buf,
a_scale_thread_desc,
```

**`include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_v3_ab_scale.hpp`**
```
constexpr auto mfma_cycle = NPerXDL == 16 ? 16 : 32;
constexpr auto ds_read_a_issue_cycle =
HotLoopInstList::A_LDS_Read_Width * sizeof(ADataType) == 16 ? 8 : 4;
constexpr auto ds_read_b_issue_cycle =
```

**`include/ck/tensor_operation/gpu/device/impl/device_gemm_multiple_d_xdl_cshuffle_v3_ab_scale.hpp`**
```
if(stream_config.flush_cache)
Argument arg_ = arg;
const auto a_grid_desc_ak0_m_ak1 = GridwiseGemm::MakeAGridDescriptor_AK0_M_AK1(
arg_.M, arg_.MPadded, arg_.K, arg_.KPadded, arg_.StrideA, arg_.AK0);
```
