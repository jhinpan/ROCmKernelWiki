# Diff summary

- **files changed:** 22
- **lines:** +43 / -24
- **kernel-ish files:** 22

## Files (by churn)

- `include/ck/tensor_operation/gpu/device/impl/device_conv2d_backward_weight_xdl_c_shuffle_nhwc_kyxc_nhwk.hpp`  (+5/-2)
- `include/ck/tensor_operation/gpu/device/impl/device_gemm_xdl_splitk_c_shuffle.hpp`  (+5/-2)
- `include/ck/tensor_operation/gpu/device/impl/device_batched_gemm_reduce_xdl_cshuffle.hpp`  (+3/-3)
- `include/ck/tensor_operation/gpu/device/impl/device_batched_gemm_softmax_gemm_permute_xdl_cshuffle.hpp`  (+1/-5)
- `include/ck/tensor_operation/gpu/device/impl/device_gemm_reduce_xdl_cshuffle.hpp`  (+3/-3)
- `include/ck/ck.hpp`  (+3/-0)
- `include/ck/tensor_operation/gpu/device/impl/device_batched_gemm_multiple_d_gemm_multiple_d_xdl_cshuffle.hpp`  (+2/-0)
- `include/ck/tensor_operation/gpu/device/impl/device_batched_gemm_xdl.hpp`  (+1/-1)
- `include/ck/tensor_operation/gpu/device/impl/device_conv2d_bwd_data_xdl_nhwc_kyxc_nhwk.hpp`  (+1/-1)
- `include/ck/tensor_operation/gpu/device/impl/device_conv2d_fwd_xdl_c_shuffle_bias_activation_add_nhwc_kyxc_nhwk.hpp`  (+1/-1)
- `include/ck/tensor_operation/gpu/device/impl/device_conv2d_fwd_xdl_c_shuffle_bias_activation_nhwc_kyxc_nhwk.hpp`  (+1/-1)
- `include/ck/tensor_operation/gpu/device/impl/device_conv2d_fwd_xdl_c_shuffle_nhwc_kyxc_nhwk.hpp`  (+1/-1)
- `include/ck/tensor_operation/gpu/device/impl/device_conv2d_fwd_xdl_nhwc_kyxc_nhwk.hpp`  (+1/-1)
- `include/ck/tensor_operation/gpu/device/impl/device_conv3d_fwd_xdl_ndhwc_kzyxc_ndhwk.hpp`  (+2/-0)
- `include/ck/tensor_operation/gpu/device/impl/device_convnd_bwd_data_nwc_kxc_nwk_dl.hpp`  (+2/-0)

## Key added lines (kernel files)

**`include/ck/tensor_operation/gpu/device/impl/device_batched_gemm_reduce_xdl_cshuffle.hpp`**
```
std::cout << "arg.reduce_grid_desc_m_{ " << arg.reduce_grid_desc_m_.GetLength(I0)
<< "}" << std::endl;
```

**`include/ck/tensor_operation/gpu/device/impl/device_conv2d_backward_weight_xdl_c_shuffle_nhwc_kyxc_nhwk.hpp`**
```
void Print(const Argument& arg)
if(stream_config.log_level_ > 0)
Print(arg);
```

**`include/ck/tensor_operation/gpu/device/impl/device_gemm_reduce_xdl_cshuffle.hpp`**
```
std::cout << "arg.reduce_grid_desc_m_{ " << arg.reduce_grid_desc_m_.GetLength(I0)
<< "}" << std::endl;
```

**`include/ck/tensor_operation/gpu/device/impl/device_gemm_xdl_splitk_c_shuffle.hpp`**
```
void Print(const Argument& arg)
if(stream_config.log_level_ > 0)
Print(arg);
```
