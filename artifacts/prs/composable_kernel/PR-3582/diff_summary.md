# Diff summary

- **files changed:** 27
- **lines:** +1007 / -171
- **kernel-ish files:** 23

## Files (by churn)

- `library/include/ck/library/tensor_operation_instance/gpu/grouped_convolution_forward_bias_clamp_wmma_cshufflev3.inc`  (+108/-56)
- `library/include/ck/library/tensor_operation_instance/gpu/grouped_convolution_forward_clamp_wmma_cshufflev3.inc`  (+108/-56)
- `include/ck/tensor_operation/gpu/device/impl/device_grouped_conv_fwd_multiple_d_wmma_cshuffle_v3_large_tensor.hpp`  (+46/-37)
- `library/include/ck/library/tensor_operation_instance/gpu/grouped_conv_fwd/device_grouped_conv_fwd_wmma_cshufflev3_large_tensor_instance.hpp`  (+46/-6)
- `library/src/tensor_operation_instance/gpu/grouped_conv2d_fwd_bias_clamp/wmma/large_tensor/device_grouped_conv2d_fwd_bias_clamp_wmma_cshufflev3_large_tensor_nhwgc_gkyxc_nhwgk_bf16_generic_instance.cpp`  (+40/-0)
- `library/src/tensor_operation_instance/gpu/grouped_conv2d_fwd_bias_clamp/wmma/large_tensor/device_grouped_conv2d_fwd_bias_clamp_wmma_cshufflev3_large_tensor_nhwgc_gkyxc_nhwgk_bf16_instance.cpp`  (+40/-0)
- `library/src/tensor_operation_instance/gpu/grouped_conv2d_fwd_bias_clamp/wmma/large_tensor/device_grouped_conv2d_fwd_bias_clamp_wmma_cshufflev3_large_tensor_nhwgc_gkyxc_nhwgk_f16_generic_instance.cpp`  (+40/-0)
- `library/src/tensor_operation_instance/gpu/grouped_conv2d_fwd_bias_clamp/wmma/large_tensor/device_grouped_conv2d_fwd_bias_clamp_wmma_cshufflev3_large_tensor_nhwgc_gkyxc_nhwgk_f16_instance.cpp`  (+40/-0)
- `library/src/tensor_operation_instance/gpu/grouped_conv2d_fwd_clamp/wmma/large_tensor/device_grouped_conv2d_fwd_clamp_wmma_cshufflev3_large_tensor_nhwgc_gkyxc_nhwgk_bf16_generic_instance.cpp`  (+40/-0)
- `library/src/tensor_operation_instance/gpu/grouped_conv2d_fwd_clamp/wmma/large_tensor/device_grouped_conv2d_fwd_clamp_wmma_cshufflev3_large_tensor_nhwgc_gkyxc_nhwgk_bf16_instance.cpp`  (+40/-0)
- `library/src/tensor_operation_instance/gpu/grouped_conv2d_fwd_clamp/wmma/large_tensor/device_grouped_conv2d_fwd_clamp_wmma_cshufflev3_large_tensor_nhwgc_gkyxc_nhwgk_f16_generic_instance.cpp`  (+40/-0)
- `library/src/tensor_operation_instance/gpu/grouped_conv2d_fwd_clamp/wmma/large_tensor/device_grouped_conv2d_fwd_clamp_wmma_cshufflev3_large_tensor_nhwgc_gkyxc_nhwgk_f16_instance.cpp`  (+40/-0)
- `library/src/tensor_operation_instance/gpu/grouped_conv3d_fwd_bias_clamp/wmma/large_tensor/device_grouped_conv3d_fwd_bias_clamp_wmma_cshufflev3_large_tensor_ndhwgc_gkzyxc_ndhwgk_bf16_generic_instance.cpp`  (+40/-0)
- `library/src/tensor_operation_instance/gpu/grouped_conv3d_fwd_bias_clamp/wmma/large_tensor/device_grouped_conv3d_fwd_bias_clamp_wmma_cshufflev3_large_tensor_ndhwgc_gkzyxc_ndhwgk_bf16_instance.cpp`  (+40/-0)
- `library/src/tensor_operation_instance/gpu/grouped_conv3d_fwd_bias_clamp/wmma/large_tensor/device_grouped_conv3d_fwd_bias_clamp_wmma_cshufflev3_large_tensor_ndhwgc_gkzyxc_ndhwgk_f16_generic_instance.cpp`  (+40/-0)

## Key added lines (kernel files)

**`include/ck/tensor_operation/gpu/device/impl/device_grouped_conv_fwd_multiple_d_wmma_cshuffle_v3_large_tensor.hpp`**
```
const auto ds_desc_mblock_mperblock_nblock_nperblock =
const auto e_desc_mblock_mperblock_nblock_nperblock =
gemm_desc_kernel_args_.Emplace(
valid_gemms_count_,
```

**`include/ck/utility/array.hpp`**
```
template <typename... Args>
__host__ constexpr auto Emplace(index_t i, Args&&... args)
-> std::enable_if_t<std::is_nothrow_constructible_v<TData, Args&&...>>
assert(i >= 0 && i < NSize);
```

**`library/include/ck/library/tensor_operation_instance/gpu/grouped_conv_fwd/device_grouped_conv_fwd_wmma_cshufflev3_large_tensor_instance.hpp`**
```
using AddClamp    = ck::tensor_operation::element_wise::AddClamp;
using Clamp       = ck::tensor_operation::element_wise::Clamp;
template <index_t NDSpatial,
typename ALayout,
```

**`library/include/ck/library/tensor_operation_instance/gpu/grouped_convolution_forward_bias_clamp.hpp`**
```
add_device_grouped_conv2d_fwd_bias_clamp_wmma_cshufflev3_large_tensor_nhwgc_gkyxc_nhwgk_bf16_instances(
op_ptrs);
add_device_grouped_conv2d_fwd_bias_clamp_wmma_cshufflev3_large_tensor_nhwgc_gkyxc_nhwgk_bf16_generic_instances(
op_ptrs);
```

**`library/include/ck/library/tensor_operation_instance/gpu/grouped_convolution_forward_bias_clamp_wmma_cshufflev3.inc`**
```
void add_device_grouped_conv2d_fwd_bias_clamp_wmma_cshufflev3_large_tensor_nhwgc_gkyxc_nhwgk_bf16_instances(
std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<2,
Tuple<NHWGK>,
Tuple<BF16>,
```
