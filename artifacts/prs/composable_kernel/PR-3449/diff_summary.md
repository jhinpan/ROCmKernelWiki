# Diff summary

- **files changed:** 46 (diff was byte-capped; summary is partial)
- **lines:** +2420 / -450
- **kernel-ish files:** 42

## Files (by churn)

- `library/include/ck/library/tensor_operation_instance/gpu/grouped_convolution_forward_bias_clamp_wmma_cshufflev3.inc`  (+160/-4)
- `library/include/ck/library/tensor_operation_instance/gpu/grouped_convolution_forward_clamp_wmma_cshufflev3.inc`  (+160/-4)
- `library/include/ck/library/tensor_operation_instance/gpu/grouped_convolution_forward_wmma_cshufflev3.inc`  (+160/-4)
- `library/include/ck/library/tensor_operation_instance/gpu/grouped_conv_fwd/device_grouped_conv_fwd_wmma_cshufflev3_instance.hpp`  (+125/-14)
- `library/src/tensor_operation_instance/gpu/grouped_conv2d_fwd/wmma/device_grouped_conv2d_fwd_wmma_cshufflev3_nhwgc_gkyxc_nhwgk_bf16_instance.cpp`  (+0/-65)
- `library/src/tensor_operation_instance/gpu/grouped_conv2d_fwd/wmma/device_grouped_conv2d_fwd_wmma_cshufflev3_nhwgc_gkyxc_nhwgk_bf16_instance_part1.cpp`  (+65/-0)
- `library/src/tensor_operation_instance/gpu/grouped_conv2d_fwd/wmma/device_grouped_conv2d_fwd_wmma_cshufflev3_nhwgc_gkyxc_nhwgk_bf16_instance_part2.cpp`  (+65/-0)
- `library/src/tensor_operation_instance/gpu/grouped_conv2d_fwd/wmma/device_grouped_conv2d_fwd_wmma_cshufflev3_nhwgc_gkyxc_nhwgk_bf16_instance_part3.cpp`  (+65/-0)
- `library/src/tensor_operation_instance/gpu/grouped_conv2d_fwd/wmma/device_grouped_conv2d_fwd_wmma_cshufflev3_nhwgc_gkyxc_nhwgk_bf16_instance_part4.cpp`  (+65/-0)
- `library/src/tensor_operation_instance/gpu/grouped_conv2d_fwd/wmma/device_grouped_conv2d_fwd_wmma_cshufflev3_nhwgc_gkyxc_nhwgk_f16_instance.cpp`  (+0/-65)
- `library/src/tensor_operation_instance/gpu/grouped_conv2d_fwd/wmma/device_grouped_conv2d_fwd_wmma_cshufflev3_nhwgc_gkyxc_nhwgk_f16_instance_part1.cpp`  (+65/-0)
- `library/src/tensor_operation_instance/gpu/grouped_conv2d_fwd/wmma/device_grouped_conv2d_fwd_wmma_cshufflev3_nhwgc_gkyxc_nhwgk_f16_instance_part2.cpp`  (+65/-0)
- `library/src/tensor_operation_instance/gpu/grouped_conv2d_fwd/wmma/device_grouped_conv2d_fwd_wmma_cshufflev3_nhwgc_gkyxc_nhwgk_f16_instance_part3.cpp`  (+65/-0)
- `library/src/tensor_operation_instance/gpu/grouped_conv2d_fwd/wmma/device_grouped_conv2d_fwd_wmma_cshufflev3_nhwgc_gkyxc_nhwgk_f16_instance_part4.cpp`  (+65/-0)
- `library/src/tensor_operation_instance/gpu/grouped_conv2d_fwd_bias_clamp/wmma/device_grouped_conv2d_fwd_bias_clamp_wmma_cshufflev3_nhwgc_gkyxc_nhwgk_bf16_instance.cpp`  (+0/-62)

## Key added lines (kernel files)

**`library/include/ck/library/tensor_operation_instance/gpu/grouped_conv_fwd/device_grouped_conv_fwd_wmma_cshufflev3_instance.hpp`**
```
using device_grouped_conv_fwd_wmma_cshufflev3_bf16_instances_part1 = std::tuple<
DeviceGroupedConvFwdMultipleABD_Wmma_CShuffle_V3<NDimSpatial, ALayout, BLayout,    DsLayout, ELayout,  BF16,  BF16,     
template <index_t NDimSpatial,
typename ALayout,
```

**`library/include/ck/library/tensor_operation_instance/gpu/grouped_convolution_forward.hpp`**
```
add_device_grouped_conv2d_fwd_wmma_cshufflev3_nhwgc_gkyxc_nhwgk_f16_instances_part1(
op_ptrs);
add_device_grouped_conv2d_fwd_wmma_cshufflev3_nhwgc_gkyxc_nhwgk_f16_instances_part2(
op_ptrs);
```

**`library/include/ck/library/tensor_operation_instance/gpu/grouped_convolution_forward_bias_clamp.hpp`**
```
add_device_grouped_conv2d_fwd_bias_clamp_wmma_cshufflev3_nhwgc_gkyxc_nhwgk_bf16_instances_part1(
op_ptrs);
add_device_grouped_conv2d_fwd_bias_clamp_wmma_cshufflev3_nhwgc_gkyxc_nhwgk_bf16_instances_part2(
op_ptrs);
```

**`library/include/ck/library/tensor_operation_instance/gpu/grouped_convolution_forward_bias_clamp_wmma_cshufflev3.inc`**
```
void add_device_grouped_conv2d_fwd_bias_clamp_wmma_cshufflev3_nhwgc_gkyxc_nhwgk_bf16_instances_part1(
std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<2,
Tuple<NHWGK>,
Tuple<BF16>,
```

**`library/include/ck/library/tensor_operation_instance/gpu/grouped_convolution_forward_clamp.hpp`**
```
add_device_grouped_conv2d_fwd_clamp_wmma_cshufflev3_nhwgc_gkyxc_nhwgk_bf16_instances_part1(
op_ptrs);
add_device_grouped_conv2d_fwd_clamp_wmma_cshufflev3_nhwgc_gkyxc_nhwgk_bf16_instances_part2(
op_ptrs);
```
