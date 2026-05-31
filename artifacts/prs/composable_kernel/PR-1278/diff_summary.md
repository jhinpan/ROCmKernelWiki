# Diff summary

- **files changed:** 31
- **lines:** +650 / -442
- **kernel-ish files:** 31

## Files (by churn)

- `include/ck/utility/env.hpp`  (+185/-0)
- `include/ck/tensor_operation/gpu/grid/gridwise_gemm_xdlops_v2r4r2.hpp`  (+70/-68)
- `include/ck/tensor_operation/gpu/grid/gridwise_gemm_xdl_cshuffle_v3.hpp`  (+69/-67)
- `include/ck/tensor_operation/gpu/grid/gridwise_gemm_xdl_cshuffle_v3_multi_abd.hpp`  (+64/-62)
- `include/ck/host_utility/kernel_launch.hpp`  (+34/-30)
- `include/ck/tensor_operation/gpu/device/impl/device_grouped_gemm_multiple_d_splitk_xdl_cshuffle_two_stage.hpp`  (+32/-28)
- `include/ck/tensor_operation/gpu/device/impl/device_batched_gemm_multiple_d_gemm_multiple_d_xdl_cshuffle.hpp`  (+25/-24)
- `include/ck/host_utility/flush_cache.hpp`  (+24/-21)
- `include/ck/tensor_operation/gpu/device/impl/device_grouped_gemm_xdl.hpp`  (+23/-22)
- `include/ck/tensor_operation/gpu/device/impl/device_grouped_gemm_multiple_d_dl.hpp`  (+23/-18)
- `include/ck/tensor_operation/gpu/device/impl/device_conv2d_bwd_data_xdl_nhwc_kyxc_nhwk.hpp`  (+19/-18)
- `include/ck/tensor_operation/gpu/device/impl/device_batched_gemm_reduce_xdl_cshuffle.hpp`  (+16/-15)
- `include/ck/tensor_operation/gpu/device/impl/device_grouped_gemm_xdl_splitk_cshuffle.hpp`  (+12/-10)
- `include/ck/tensor_operation/gpu/device/impl/device_grouped_gemm_multiple_d_xdl_cshuffle_tile_loop.hpp`  (+7/-5)
- `profiler/include/profiler/profile_grouped_gemm_fixed_nk_impl.hpp`  (+6/-5)

## Key added lines (kernel files)

**`include/ck/ck.hpp`**
```
CK_DECLARE_ENV_VAR_BOOL(CK_LOGGING)
```

**`include/ck/host_utility/flush_cache.hpp`**
```
if(ck::EnvIsEnabled(ENV(CK_LOGGING)))
printf("%s: grid_dim {%d, %d, %d}, block_dim {%d, %d, %d} \n",
__func__,
grid_dim.x,
```

**`include/ck/host_utility/kernel_launch.hpp`**
```
if(ck::EnvIsEnabled(ENV(CK_LOGGING)))
printf("%s: grid_dim {%d, %d, %d}, block_dim {%d, %d, %d} \n",
__func__,
grid_dim.x,
```

**`include/ck/tensor_operation/gpu/device/impl/device_batched_gemm_multiple_d_gemm_multiple_d_xdl_cshuffle.hpp`**
```
if(ck::EnvIsEnabled(ENV(CK_LOGGING)))
std::cout << "a0_grid_desc_m_k_{" << a0_grid_desc_m_k_.GetLength(I0) << ", "
<< a0_grid_desc_m_k_.GetLength(I1) << "}" << std::endl;
std::cout << "b0_grid_desc_n_k_{" << b0_grid_desc_n_k_.GetLength(I0) << ", "
```

**`include/ck/tensor_operation/gpu/device/impl/device_batched_gemm_reduce_xdl_cshuffle.hpp`**
```
if(ck::EnvIsEnabled(ENV(CK_LOGGING)))
std::cout << "arg.Batch_ = " << arg.Batch_ << std::endl;
std::cout << "arg.a_grid_desc_ak0_m_ak1_{"
<< arg.a_grid_desc_ak0_m_ak1_.GetLength(I0) << ", "
```
