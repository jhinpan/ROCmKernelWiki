# Diff summary

- **files changed:** 37
- **lines:** +121 / -51
- **kernel-ish files:** 37

## Files (by churn)

- `include/ck/host_utility/device_prop.hpp`  (+7/-0)
- `example/01_gemm/gemm_xdl_skip_b_lds_fp16.cpp`  (+3/-3)
- `include/ck/tensor_operation/gpu/device/impl/device_batched_gemm_e_permute_xdl.hpp`  (+5/-0)
- `include/ck/tensor_operation/gpu/device/impl/device_batched_gemm_reduce_xdl_cshuffle.hpp`  (+5/-0)
- `include/ck/tensor_operation/gpu/device/impl/device_batched_gemm_xdl.hpp`  (+5/-0)
- `include/ck/tensor_operation/gpu/device/impl/device_cgemm_4gemm_xdl_cshuffle.hpp`  (+5/-0)
- `include/ck/tensor_operation/gpu/device/impl/device_conv2d_backward_weight_xdl_c_shuffle_nhwc_kyxc_nhwk.hpp`  (+5/-0)
- `include/ck/tensor_operation/gpu/device/impl/device_conv2d_bwd_data_xdl_nhwc_kyxc_nhwk.hpp`  (+5/-0)
- `include/ck/tensor_operation/gpu/device/impl/device_conv2d_fwd_xdl_c_shuffle_bias_activation_add_nhwc_kyxc_nhwk.hpp`  (+5/-0)
- `include/ck/tensor_operation/gpu/device/impl/device_conv2d_fwd_xdl_c_shuffle_bias_activation_nhwc_kyxc_nhwk.hpp`  (+5/-0)
- `include/ck/tensor_operation/gpu/device/impl/device_conv2d_fwd_xdl_c_shuffle_nhwc_kyxc_nhwk.hpp`  (+5/-0)
- `include/ck/tensor_operation/gpu/device/impl/device_conv2d_fwd_xdl_nhwc_kyxc_nhwk.hpp`  (+5/-0)
- `include/ck/tensor_operation/gpu/device/impl/device_conv3d_fwd_xdl_ndhwc_kzyxc_ndhwk.hpp`  (+5/-0)
- `include/ck/tensor_operation/gpu/device/impl/device_convnd_bwd_data_nwc_kxc_nwk_xdl.hpp`  (+5/-0)
- `include/ck/tensor_operation/gpu/device/impl/device_gemm_bias_add_reduce_xdl_cshuffle.hpp`  (+5/-0)

## Key added lines (kernel files)

**`example/01_gemm/gemm_xdl_skip_b_lds_fp16.cpp`**
```
std::cerr << gemm.GetTypeString() << " does not support this problem" << std::endl;
return 0;
```

**`include/ck/host_utility/device_prop.hpp`**
```
inline bool is_xdl_supported()
return ck::get_device_name() == "gfx908" || ck::get_device_name() == "gfx90a" ||
ck::get_device_name() == "gfx940" || ck::get_device_name() == "gfx941" ||
ck::get_device_name() == "gfx942";
```

**`include/ck/tensor_operation/gpu/device/impl/device_batched_contraction_multiple_d_xdl_cshuffle.hpp`**
```
if(!ck::is_xdl_supported())
```

**`include/ck/tensor_operation/gpu/device/impl/device_batched_gemm_e_permute_xdl.hpp`**
```
if(!ck::is_xdl_supported())
return false;
```

**`include/ck/tensor_operation/gpu/device/impl/device_batched_gemm_gemm_xdl_cshuffle.hpp`**
```
if(!ck::is_xdl_supported())
```
