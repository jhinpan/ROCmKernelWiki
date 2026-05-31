# Diff summary

- **files changed:** 44
- **lines:** +175 / -1085
- **kernel-ish files:** 37

## Files (by churn)

- `include/ck/tensor_operation/gpu/warp/xdlops_gemm.hpp`  (+22/-114)
- `example/09_convnd_fwd/convnd_fwd_xdl_fp32_tf32.cpp`  (+0/-89)
- `example/01_gemm/gemm_xdl_lds_direct_load_fp32_tf32.cpp`  (+0/-85)
- `include/ck/tensor_operation/gpu/block/blockwise_gemm_xdlops.hpp`  (+34/-51)
- `library/src/tensor_operation_instance/gpu/grouped_conv3d_fwd_bias_bnorm_clamp/xdl/device_grouped_conv3d_fwd_bias_bn_clamp_xdl_ndhwgc_gkzyxc_ndhwgk_f32_tf32_instance.in`  (+0/-81)
- `library/src/tensor_operation_instance/gpu/grouped_conv3d_fwd_bias_clamp/xdl/device_grouped_conv3d_fwd_bias_clamp_xdl_ndhwgc_gkzyxc_ndhwgk_fp32_tf32_instance.cpp`  (+0/-60)
- `library/src/tensor_operation_instance/gpu/grouped_conv3d_fwd_clamp/xdl/device_grouped_conv3d_fwd_clamp_xdl_ndhwgc_gkzyxc_ndhwgk_fp32_tf32_instance.cpp`  (+0/-60)
- `library/src/tensor_operation_instance/gpu/grouped_conv3d_fwd_bias_bnorm_clamp/CMakeLists.txt`  (+25/-34)
- `library/src/tensor_operation_instance/gpu/grouped_conv3d_fwd/xdl/device_grouped_conv3d_fwd_xdl_ndhwgc_gkzyxc_ndhwgk_f32_tf32_instance.cpp`  (+0/-56)
- `include/ck/tensor_operation/gpu/device/impl/device_grouped_conv_fwd_multiple_abd_xdl_cshuffle.hpp`  (+3/-48)
- `library/include/ck/library/tensor_operation_instance/gpu/grouped_conv_fwd/device_grouped_conv_fwd_xdl_instance.hpp`  (+2/-41)
- `include/ck/utility/amd_xdlops.hpp`  (+0/-41)
- `library/include/ck/library/reference_tensor_operation/cpu/reference_gemm.hpp`  (+12/-28)
- `include/ck/tensor_operation/gpu/grid/gridwise_gemm_multiple_d_xdl_cshuffle.hpp`  (+18/-21)
- `include/ck/utility/data_type.hpp`  (+0/-35)

## Key added lines (kernel files)

**`example/01_gemm/common.hpp`**
```
template <typename DataType>
if constexpr(std::is_same_v<DataType, float>)
template <typename DataType>
if constexpr(std::is_same_v<DataType, float>)
```

**`example/01_gemm/run_gemm_example.inc`**
```
get_rtol<CDataType>(),
get_atol<CDataType>());
get_rtol<CDataType>(),
get_atol<CDataType>());
```

**`example/09_convnd_fwd/convnd_fwd_common.hpp`**
```
template <typename DataType>
if constexpr(std::is_same_v<DataType, float>)
template <typename DataType>
if constexpr(std::is_same_v<DataType, float>)
```

**`example/09_convnd_fwd/run_convnd_fwd_example.inc`**
```
DeviceGroupedConvNDFwdInstance<ndim_spatial_value, InLayout, WeiLayout, OutLayout>>(
do_verification,
init_method,
time_kernel,
```

**`include/ck/library/utility/check_err.hpp`**
```
err_count++;
```
