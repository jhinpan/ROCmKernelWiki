# Diff summary

- **files changed:** 42
- **lines:** +64 / -31
- **kernel-ish files:** 42

## Files (by churn)

- `include/ck_tile/core/config.hpp`  (+0/-6)
- `include/ck/ck.hpp`  (+0/-5)
- `include/ck/utility/env.hpp`  (+5/-0)
- `include/ck_tile/core/utility/env.hpp`  (+4/-0)
- `include/ck/tensor_operation/gpu/device/impl/device_batched_gemm_reduce_xdl_cshuffle.hpp`  (+2/-1)
- `include/ck/tensor_operation/gpu/device/impl/device_batched_gemm_softmax_gemm_permute_xdl_cshuffle.hpp`  (+2/-1)
- `include/ck/tensor_operation/gpu/device/impl/device_conv2d_bwd_data_xdl_nhwc_kyxc_nhwk.hpp`  (+2/-1)
- `include/ck/tensor_operation/gpu/device/impl/device_conv2d_fwd_xdl_c_shuffle_bias_activation_add_nhwc_kyxc_nhwk.hpp`  (+2/-1)
- `include/ck/tensor_operation/gpu/device/impl/device_conv2d_fwd_xdl_c_shuffle_bias_activation_nhwc_kyxc_nhwk.hpp`  (+2/-1)
- `include/ck/tensor_operation/gpu/device/impl/device_conv2d_fwd_xdl_c_shuffle_nhwc_kyxc_nhwk.hpp`  (+2/-1)
- `include/ck/tensor_operation/gpu/device/impl/device_conv2d_fwd_xdl_nhwc_kyxc_nhwk.hpp`  (+2/-1)
- `include/ck/tensor_operation/gpu/device/impl/device_conv3d_fwd_xdl_ndhwc_kzyxc_ndhwk.hpp`  (+2/-1)
- `include/ck/tensor_operation/gpu/device/impl/device_convnd_bwd_data_nwc_kxc_nwk_xdl.hpp`  (+2/-1)
- `include/ck/tensor_operation/gpu/device/impl/device_gemm_dl.hpp`  (+2/-1)
- `include/ck/tensor_operation/gpu/device/impl/device_gemm_xdl_layernorm_cshuffle.hpp`  (+2/-1)

## Key added lines (kernel files)

**`include/ck/utility/env.hpp`**
```
CK_DECLARE_ENV_VAR_BOOL(CK_LOGGING)
```

**`include/ck_tile/core/utility/env.hpp`**
```
CK_TILE_DECLARE_ENV_VAR_BOOL(CK_TILE_LOGGING)
```
