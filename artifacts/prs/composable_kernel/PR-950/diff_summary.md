# Diff summary

- **files changed:** 47
- **lines:** +2272 / -214
- **kernel-ish files:** 43

## Files (by churn)

- `include/ck/tensor_operation/gpu/device/impl/device_grouped_conv_bwd_data_multiple_d_wmma_cshuffle.hpp`  (+879/-0)
- `library/include/ck/library/tensor_operation_instance/gpu/grouped_convolution_backward_data.hpp`  (+283/-0)
- `test/grouped_convnd_bwd_data/test_grouped_convnd_bwd_data_interface_wmma.cpp`  (+178/-0)
- `library/include/ck/library/tensor_operation_instance/gpu/grouped_conv_bwd_data/device_grouped_conv_bwd_data_wmma_instance.hpp`  (+118/-0)
- `include/ck/tensor_operation/gpu/device/impl/device_grouped_conv_utils.hpp`  (+59/-0)
- `include/ck/tensor_operation/gpu/device/impl/device_grouped_conv_bwd_data_multiple_d_xdl_cshuffle_v1.hpp`  (+3/-47)
- `include/ck/tensor_operation/gpu/device/impl/device_grouped_conv_fwd_multiple_d_wmma_cshuffle.hpp`  (+1/-49)
- `include/ck/tensor_operation/gpu/device/impl/device_grouped_conv_fwd_dl_multiple_d_nhwc_kyxc_nhwk.hpp`  (+1/-45)
- `include/ck/tensor_operation/gpu/device/impl/device_grouped_conv_fwd_multiple_d_xdl_cshuffle.hpp`  (+1/-45)
- `library/src/tensor_operation_instance/gpu/grouped_conv2d_bwd_data/wmma/device_grouped_conv2d_bwd_data_wmma_gnhwc_gkyxc_gnhwk_f16_1x1s1p0_instance.cpp`  (+40/-0)
- `library/src/tensor_operation_instance/gpu/grouped_conv2d_bwd_data/wmma/device_grouped_conv2d_bwd_data_wmma_gnhwc_gkyxc_gnhwk_f16_instance.cpp`  (+40/-0)
- `library/src/tensor_operation_instance/gpu/grouped_conv2d_bwd_data/wmma/device_grouped_conv2d_bwd_data_wmma_gnhwc_gkyxc_gnhwk_i8_1x1s1p0_instance.cpp`  (+40/-0)
- `library/src/tensor_operation_instance/gpu/grouped_conv2d_bwd_data/wmma/device_grouped_conv2d_bwd_data_wmma_gnhwc_gkyxc_gnhwk_i8_instance.cpp`  (+40/-0)
- `library/src/tensor_operation_instance/gpu/grouped_conv2d_bwd_data/wmma/device_grouped_conv2d_bwd_data_wmma_nhwgc_gkyxc_nhwgk_f16_1x1s1p0_instance.cpp`  (+40/-0)
- `library/src/tensor_operation_instance/gpu/grouped_conv2d_bwd_data/wmma/device_grouped_conv2d_bwd_data_wmma_nhwgc_gkyxc_nhwgk_f16_instance.cpp`  (+40/-0)

## Key added lines (kernel files)

**`example/38_grouped_conv_bwd_data_multiple_d/grouped_conv_bwd_data_wmma_fp16.cpp`**
```
using OutDataType      = FP16;
using WeiDataType      = FP16;
using AccDataType      = FP32;
using CShuffleDataType = FP16;
```

**`include/ck/tensor_operation/gpu/device/impl/device_grouped_conv_bwd_data_multiple_d_wmma_cshuffle.hpp`**
```
namespace ck {
namespace tensor_operation {
namespace device {
template <index_t NDimSpatial,
```

**`include/ck/tensor_operation/gpu/device/impl/device_grouped_conv_utils.hpp`**
```
namespace ck {
namespace tensor_operation {
namespace device {
template <index_t NumDTensor>
```

**`library/include/ck/library/tensor_operation_instance/gpu/grouped_conv_bwd_data/device_grouped_conv_bwd_data_wmma_instance.hpp`**
```
namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {
```

**`library/include/ck/library/tensor_operation_instance/gpu/grouped_convolution_backward_data.hpp`**
```
void add_device_grouped_conv2d_bwd_data_wmma_gnhwk_gkyxc_gnhwc_f16_instances(
std::vector<std::unique_ptr<DeviceGroupedConvBwdDataMultipleD<2,
Empty_Tuple,
Empty_Tuple,
```
