# Diff summary

- **files changed:** 18
- **lines:** +662 / -1017
- **kernel-ish files:** 16

## Files (by churn)

- `include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_v1_ab_scale.hpp`  (+83/-532)
- `include/ck/tensor_operation/gpu/grid/gridwise_gemm_xdl_cshuffle_v3_multi_d_ab_scale.hpp`  (+123/-105)
- `include/ck/tensor_operation/gpu/device/impl/device_gemm_multiple_d_xdl_cshuffle_v3_ab_scale.hpp`  (+128/-69)
- `include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_v3_ab_scale.hpp`  (+33/-120)
- `include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_v2_ab_scale.hpp`  (+25/-68)
- `library/include/ck/library/tensor_operation_instance/gpu/gemm_ab_scale.hpp`  (+74/-14)
- `example/65_gemm_multiply_multiply/gemm_multiply_multiply_xdl_fp8_ab_scale.cpp`  (+20/-52)
- `library/src/tensor_operation_instance/gpu/gemm_ab_scale/device_gemm_ab_scale_xdl_f8_f8_bf16/device_gemm_ab_scale_xdl_f8_f8_bf16_mk_nk_mn_128_128_128.hpp`  (+35/-36)
- `library/src/tensor_operation_instance/gpu/gemm_ab_scale/device_gemm_ab_scale_xdl_f8_f8_bf16/device_gemm_ab_scale_xdl_f8_f8_bf16_mk_nk_mn_128_128_128_mem_v1_mnkpadding_instance.cpp`  (+38/-0)
- `library/src/tensor_operation_instance/gpu/gemm_ab_scale/device_gemm_ab_scale_xdl_f8_f8_bf16/device_gemm_ab_scale_xdl_f8_f8_bf16_mk_nk_mn_128_128_128_comp_mnkpadding_instance.cpp`  (+37/-0)
- `library/src/tensor_operation_instance/gpu/gemm_ab_scale/device_gemm_ab_scale_xdl_f8_f8_bf16/device_gemm_ab_scale_xdl_f8_f8_bf16_mk_nk_mn_128_128_128_comp_mnpadding_instance.cpp`  (+37/-0)
- `library/src/tensor_operation_instance/gpu/gemm_ab_scale/device_gemm_ab_scale_xdl_f8_f8_bf16/device_gemm_ab_scale_xdl_f8_f8_bf16_mk_nk_mn_128_128_128_mem_v1_default_instance.cpp`  (+4/-4)
- `library/src/tensor_operation_instance/gpu/gemm_ab_scale/device_gemm_ab_scale_xdl_f8_f8_bf16/device_gemm_ab_scale_xdl_f8_f8_bf16_mk_nk_mn_128_128_128_mem_v1_kpadding_instance.cpp`  (+4/-4)
- `profiler/src/profile_gemm_ab_scale.cpp`  (+3/-5)
- `CMakeLists.txt`  (+7/-0)

## Key added lines (kernel files)

**`example/65_gemm_multiply_multiply/gemm_multiply_multiply_xdl_fp8_ab_scale.cpp`**
```
static constexpr ck::index_t Scale_Block_M = 128;
128, 128,
128, 16, 16,
S<8, 32, 1>, S<1, 0, 2>, S<1, 0, 2>, 2, 16, 16, 0,
```

**`include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_v1_ab_scale.hpp`**
```
using Base::AMmaKStride;
using Base::BMmaKStride;
static constexpr index_t PrefetchStages  = 1;
ignore = num_loop;
```

**`include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_v2_ab_scale.hpp`**
```
a_scale_thread_copy.Run(a_scale_grid_desc,
a_scale_grid_buf,
a_scale_thread_desc,
make_tuple(I0, I0),
```

**`include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_v3_ab_scale.hpp`**
```
constexpr auto mfma_cycle            = NPerXDL == 16 ? 16 : 32;
constexpr auto ds_read_a_issue_cycle = 4;
constexpr auto ds_read_b_issue_cycle = 4;
index_t num_loop,
```

**`include/ck/tensor_operation/gpu/device/impl/device_gemm_multiple_d_xdl_cshuffle_v3_ab_scale.hpp`**
```
if(arg.KBatch > 1)
hipGetErrorString(hipMemsetAsync(arg.p_c_grid,
arg.M * arg.N * sizeof(CDataType),
stream_config.stream_id_));
```
