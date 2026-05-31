# Diff summary

- **files changed:** 45
- **lines:** +1147 / -181
- **kernel-ish files:** 37

## Files (by churn)

- `include/ck/tensor_operation/gpu/warp/xdlops_gemm.hpp`  (+114/-22)
- `example/09_convnd_fwd/convnd_fwd_xdl_fp32_tf32.cpp`  (+89/-0)
- `example/01_gemm/gemm_xdl_lds_direct_load_fp32_tf32.cpp`  (+85/-0)
- `include/ck/tensor_operation/gpu/block/blockwise_gemm_xdlops.hpp`  (+51/-34)
- `library/src/tensor_operation_instance/gpu/grouped_conv3d_fwd_bias_bnorm_clamp/xdl/device_grouped_conv3d_fwd_bias_bn_clamp_xdl_ndhwgc_gkzyxc_ndhwgk_f32_tf32_instance.in`  (+81/-0)
- `library/src/tensor_operation_instance/gpu/grouped_conv3d_fwd_bias_clamp/xdl/device_grouped_conv3d_fwd_bias_clamp_xdl_ndhwgc_gkzyxc_ndhwgk_fp32_tf32_instance.cpp`  (+60/-0)
- `library/src/tensor_operation_instance/gpu/grouped_conv3d_fwd_clamp/xdl/device_grouped_conv3d_fwd_clamp_xdl_ndhwgc_gkzyxc_ndhwgk_fp32_tf32_instance.cpp`  (+60/-0)
- `library/src/tensor_operation_instance/gpu/grouped_conv3d_fwd_bias_bnorm_clamp/CMakeLists.txt`  (+34/-25)
- `library/src/tensor_operation_instance/gpu/grouped_conv3d_fwd/xdl/device_grouped_conv3d_fwd_xdl_ndhwgc_gkzyxc_ndhwgk_f32_tf32_instance.cpp`  (+56/-0)
- `include/ck/tensor_operation/gpu/device/impl/device_grouped_conv_fwd_multiple_abd_xdl_cshuffle.hpp`  (+48/-3)
- `library/include/ck/library/reference_tensor_operation/cpu/reference_conv_fwd.hpp`  (+37/-6)
- `library/include/ck/library/tensor_operation_instance/gpu/grouped_conv_fwd/device_grouped_conv_fwd_xdl_instance.hpp`  (+41/-2)
- `include/ck/utility/amd_xdlops.hpp`  (+41/-0)
- `profiler/src/profile_grouped_conv_fwd.cpp`  (+32/-9)
- `library/include/ck/library/reference_tensor_operation/cpu/reference_gemm.hpp`  (+28/-12)

## Key added lines (kernel files)

**`example/01_gemm/common.hpp`**
```
template <typename DataType, typename ComputeDataType = DataType>
if constexpr(std::is_same_v<DataType, float> && std::is_same_v<ComputeDataType, ck::tf32_t>)
return 1e-3;
else if constexpr(std::is_same_v<DataType, float>)
```

**`example/01_gemm/gemm_xdl_lds_direct_load_fp32_tf32.cpp`**
```
using F32 = float;
using ADataType        = F32;
using BDataType        = F32;
using AccDataType      = F32;
```

**`example/01_gemm/run_gemm_example.inc`**
```
using ComputeDataType = AccDataType;
get_rtol<CDataType, ComputeDataType>(),
get_atol<CDataType, ComputeDataType>());
get_rtol<CDataType, ComputeDataType>(),
```

**`example/09_convnd_fwd/convnd_fwd_common.hpp`**
```
template <typename DataType, typename GemmType = DataType>
if constexpr(std::is_same_v<DataType, float> && std::is_same_v<GemmType, ck::tf32_t>)
return 5e-3;
else if constexpr(std::is_same_v<DataType, float>)
```

**`example/09_convnd_fwd/convnd_fwd_xdl_fp32_tf32.cpp`**
```
using InDataType       = float;
using WeiDataType      = float;
using AccDataType      = float;
using CShuffleDataType = float;
```
