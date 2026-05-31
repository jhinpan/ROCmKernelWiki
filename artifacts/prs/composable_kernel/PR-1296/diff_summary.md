# Diff summary

- **files changed:** 30
- **lines:** +65 / -65
- **kernel-ish files:** 30

## Files (by churn)

- `include/ck/tensor_operation/gpu/grid/gridwise_gemm_xdl_cshuffle_v3.hpp`  (+10/-10)
- `include/ck/tensor_operation/gpu/grid/gridwise_gemm_xdlops_v2r4r2.hpp`  (+10/-10)
- `include/ck/tensor_operation/gpu/grid/gridwise_gemm_xdl_cshuffle_v3_multi_abd.hpp`  (+9/-9)
- `include/ck/host_utility/kernel_launch.hpp`  (+4/-4)
- `include/ck/tensor_operation/gpu/device/impl/device_grouped_gemm_multiple_d_splitk_xdl_cshuffle_two_stage.hpp`  (+4/-4)
- `include/ck/host_utility/flush_cache.hpp`  (+3/-3)
- `include/ck/tensor_operation/gpu/device/impl/device_grouped_gemm_xdl_splitk_cshuffle.hpp`  (+2/-2)
- `include/ck/tensor_operation/gpu/device/impl/device_batched_gemm_multiple_d_gemm_multiple_d_xdl_cshuffle.hpp`  (+1/-1)
- `include/ck/tensor_operation/gpu/device/impl/device_batched_gemm_reduce_xdl_cshuffle.hpp`  (+1/-1)
- `include/ck/tensor_operation/gpu/device/impl/device_batched_gemm_softmax_gemm_permute_xdl_cshuffle.hpp`  (+1/-1)
- `include/ck/tensor_operation/gpu/device/impl/device_conv2d_bwd_data_xdl_nhwc_kyxc_nhwk.hpp`  (+1/-1)
- `include/ck/tensor_operation/gpu/device/impl/device_conv2d_fwd_xdl_c_shuffle_bias_activation_add_nhwc_kyxc_nhwk.hpp`  (+1/-1)
- `include/ck/tensor_operation/gpu/device/impl/device_conv2d_fwd_xdl_c_shuffle_bias_activation_nhwc_kyxc_nhwk.hpp`  (+1/-1)
- `include/ck/tensor_operation/gpu/device/impl/device_conv2d_fwd_xdl_c_shuffle_nhwc_kyxc_nhwk.hpp`  (+1/-1)
- `include/ck/tensor_operation/gpu/device/impl/device_conv2d_fwd_xdl_nhwc_kyxc_nhwk.hpp`  (+1/-1)

## Key added lines (kernel files)

**`include/ck/host_utility/flush_cache.hpp`**
```
if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
```

**`include/ck/host_utility/kernel_launch.hpp`**
```
if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
```

**`include/ck/tensor_operation/gpu/device/impl/device_batched_gemm_multiple_d_gemm_multiple_d_xdl_cshuffle.hpp`**
```
if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
```

**`include/ck/tensor_operation/gpu/device/impl/device_batched_gemm_reduce_xdl_cshuffle.hpp`**
```
if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
```

**`include/ck/tensor_operation/gpu/device/impl/device_batched_gemm_softmax_gemm_permute_xdl_cshuffle.hpp`**
```
if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
```
