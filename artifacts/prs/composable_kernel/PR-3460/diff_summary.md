# Diff summary

- **files changed:** 26 (diff was byte-capped; summary is partial)
- **lines:** +3514 / -4
- **kernel-ish files:** 23

## Files (by churn)

- `include/ck/tensor_operation/gpu/device/impl/device_grouped_conv_bwd_data_multiple_d_wmma_cshuffle_v3.hpp`  (+1994/-0)
- `example/38_grouped_conv_bwd_data_multiple_d/run_grouped_conv3d_bwd_data_example.inc`  (+192/-0)
- `include/ck/tensor_operation/gpu/grid/gridwise_gemm_wmma_cshuffle_v3.hpp`  (+141/-0)
- `library/include/ck/library/tensor_operation_instance/gpu/grouped_conv_bwd_data/device_grouped_conv_bwd_data_wmma_v3_instances.hpp`  (+125/-0)
- `library/include/ck/library/tensor_operation_instance/gpu/grouped_convolution_backward_data_wmma.inc`  (+121/-1)
- `example/38_grouped_conv_bwd_data_multiple_d/common_conv3d.hpp`  (+116/-0)
- `library/include/ck/library/tensor_operation_instance/gpu/grouped_conv_bwd_data/device_grouped_conv_bwd_data_wmma_v3_scale_instance.hpp`  (+102/-0)
- `library/include/ck/library/tensor_operation_instance/gpu/grouped_conv_bwd_data/device_grouped_conv_bwd_data_wmma_v3_bilinear_instance.hpp`  (+100/-0)
- `library/include/ck/library/tensor_operation_instance/gpu/grouped_convolution_backward_data_bilinear.hpp`  (+71/-0)
- `library/include/ck/library/tensor_operation_instance/gpu/grouped_convolution_backward_data_scale.hpp`  (+67/-0)
- `library/src/tensor_operation_instance/gpu/grouped_conv2d_bwd_data/wmma/device_grouped_conv2d_bwd_data_wmma_v3_nhwgc_gkyxc_nhwgk_bf16_16_16_instance.cpp`  (+49/-0)
- `library/src/tensor_operation_instance/gpu/grouped_conv2d_bwd_data/wmma/device_grouped_conv2d_bwd_data_wmma_v3_nhwgc_gkyxc_nhwgk_bf16_instance.cpp`  (+49/-0)
- `library/src/tensor_operation_instance/gpu/grouped_conv2d_bwd_data/wmma/device_grouped_conv2d_bwd_data_wmma_v3_nhwgc_gkyxc_nhwgk_f16_16_16_instance.cpp`  (+49/-0)
- `library/src/tensor_operation_instance/gpu/grouped_conv2d_bwd_data/wmma/device_grouped_conv2d_bwd_data_wmma_v3_nhwgc_gkyxc_nhwgk_f16_instance.cpp`  (+49/-0)
- `example/38_grouped_conv_bwd_data_multiple_d/grouped_conv_bwd_data_wmma_v3_fp16_comp_bf8_fp8.cpp`  (+47/-0)

## Key added lines (kernel files)

**`example/38_grouped_conv_bwd_data_multiple_d/common.hpp`**
```
static constexpr auto ConvBwdDataFilter1x1Stride1Pad0 =
ck::tensor_operation::device::ConvolutionBackwardDataSpecialization::Filter1x1Stride1Pad0;
using BF16 = ck::bhalf_t;
```

**`example/38_grouped_conv_bwd_data_multiple_d/common_conv3d.hpp`**
```
using ::ck::DeviceMem;
using ::ck::hip_check_error;
using ::ck::HostTensorDescriptor;
using ::ck::Tensor;
```

**`example/38_grouped_conv_bwd_data_multiple_d/grouped_conv3d_bwd_data_wmma_v3_bf16.cpp`**
```
using OutDataType      = BF16;
using WeiDataType      = BF16;
using AccDataType      = FP32;
using CShuffleDataType = BF16;
```

**`example/38_grouped_conv_bwd_data_multiple_d/grouped_conv3d_bwd_data_wmma_v3_fp16.cpp`**
```
using OutDataType      = FP16;
using WeiDataType      = FP16;
using AccDataType      = FP32;
using CShuffleDataType = FP16;
```

**`example/38_grouped_conv_bwd_data_multiple_d/grouped_conv_bwd_data_bias_relu_wmma_v3_fp16.cpp`**
```
using OutDataType      = FP16;
using WeiDataType      = FP16;
using AccDataType      = FP32;
using CShuffleDataType = FP16;
```
