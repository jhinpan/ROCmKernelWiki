# Diff summary

- **files changed:** 28
- **lines:** +254 / -26
- **kernel-ish files:** 28

## Files (by churn)

- `include/ck/tensor_operation/gpu/device/device_grouped_conv_fwd_dl_nhwc_kyxc_nhwk.hpp`  (+13/-1)
- `include/ck/tensor_operation/gpu/device/impl/device_conv2d_backward_weight_xdl_c_shuffle_nhwc_kyxc_nhwk.hpp`  (+13/-1)
- `include/ck/tensor_operation/gpu/device/impl/device_conv2d_fwd_xdl_c_shuffle_bias_activation_add_nhwc_kyxc_nhwk.hpp`  (+13/-1)
- `include/ck/tensor_operation/gpu/device/impl/device_conv2d_fwd_xdl_c_shuffle_bias_activation_nhwc_kyxc_nhwk.hpp`  (+13/-1)
- `include/ck/tensor_operation/gpu/device/impl/device_conv3d_fwd_xdl_ndhwc_kzyxc_ndhwk.hpp`  (+13/-1)
- `include/ck/tensor_operation/gpu/device/impl/device_gemm_bias_e_permute_xdl.hpp`  (+13/-1)
- `include/ck/tensor_operation/gpu/device/impl/device_grouped_conv_fwd_multiple_d_wmma_cshuffle.hpp`  (+13/-1)
- `include/ck/tensor_operation/gpu/device/impl/device_conv2d_fwd_xdl_c_shuffle_nhwc_kyxc_nhwk.hpp`  (+11/-1)
- `include/ck/tensor_operation/gpu/device/impl/device_grouped_conv_bwd_weight_gnwc_gkxc_gnwk_xdl_cshuffle.hpp`  (+11/-1)
- `include/ck/tensor_operation/gpu/device/device_grouped_gemm_softmax_gemm_permute_xdl_cshuffle.hpp`  (+10/-1)
- `include/ck/tensor_operation/gpu/device/impl/device_conv2d_fwd_xdl_nhwc_kyxc_nhwk.hpp`  (+10/-1)
- `include/ck/tensor_operation/gpu/device/impl/device_gemm_xdl_splitk_c_shuffle.hpp`  (+10/-1)
- `include/ck/tensor_operation/gpu/device/impl/device_gemm_bias_add_reduce_xdl_cshuffle.hpp`  (+9/-1)
- `include/ck/tensor_operation/gpu/device/impl/device_gemm_multiple_d_multiple_r_xdl_cshuffle.hpp`  (+9/-1)
- `include/ck/tensor_operation/gpu/device/impl/device_gemm_reduce_xdl_cshuffle.hpp`  (+9/-1)

## Key added lines (kernel files)

**`include/ck/tensor_operation/gpu/device/device_grouped_conv_fwd_dl_multiple_d_nhwc_kyxc_nhwk.hpp`**
```
<< getConvForwardSpecializationString(ConvForwardSpecialization) << ", "
```

**`include/ck/tensor_operation/gpu/device/device_grouped_conv_fwd_dl_nhwc_kyxc_nhwk.hpp`**
```
<< getConvForwardSpecializationString(ConvForwardSpecialization) << ", "
<< K1 << ", "
<< MPerXDL << ", "
<< NPerXDL << ", "
```

**`include/ck/tensor_operation/gpu/device/device_grouped_gemm_softmax_gemm_permute_xdl_cshuffle.hpp`**
```
<< getMaskingSpecializationString(MaskingSpec) << ", "
<< MPerXDL << ", "
<< NPerXDL << ", "
<< MXdlPerWave << ", "
```

**`include/ck/tensor_operation/gpu/device/impl/device_conv2d_backward_weight_xdl_c_shuffle_nhwc_kyxc_nhwk.hpp`**
```
<< K0PerBlock << ", "
<< K1 << ", "
<< MPerXDL << ", "
<< NPerXDL << ", "
```

**`include/ck/tensor_operation/gpu/device/impl/device_conv2d_bwd_data_xdl_nhwc_kyxc_nhwk.hpp`**
```
<< K0PerBlock << ", "
<< K1 << ", "
<< MXdlPerWave << ", "
<< NXdlPerWave << ", "
```
