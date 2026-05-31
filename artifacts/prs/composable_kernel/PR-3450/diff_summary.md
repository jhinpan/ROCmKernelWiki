# Diff summary

- **files changed:** 31
- **lines:** +2568 / -26
- **kernel-ish files:** 26

## Files (by churn)

- `profiler/include/profiler/profile_grouped_conv_fwd_bilinear_impl.hpp`  (+316/-0)
- `library/include/ck/library/tensor_operation_instance/gpu/grouped_conv_fwd/device_grouped_conv_fwd_wmma_cshufflev3_bilinear_instance.hpp`  (+247/-0)
- `library/include/ck/library/tensor_operation_instance/gpu/grouped_conv_fwd/device_grouped_conv_fwd_wmma_cshufflev3_scale_instance.hpp`  (+247/-0)
- `profiler/src/profile_grouped_conv_fwd_bilinear.cpp`  (+186/-0)
- `library/include/ck/library/tensor_operation_instance/gpu/grouped_convolution_forward_bilinear.hpp`  (+154/-0)
- `library/include/ck/library/tensor_operation_instance/gpu/grouped_convolution_forward_scale.hpp`  (+141/-0)
- `test/grouped_convnd_fwd/test_grouped_convnd_fwd_bilinear.cpp`  (+134/-0)
- `test/grouped_convnd_fwd_activation/test_grouped_convnd_fwd_scale.cpp`  (+124/-0)
- `profiler/include/profiler/profile_grouped_conv_fwd_outelementop_impl.hpp`  (+62/-11)
- `library/src/tensor_operation_instance/gpu/grouped_conv3d_fwd_bilinear/wmma/device_grouped_conv3d_fwd_wmma_bilinear_ndhwgc_gkzyxc_ndhwgk_bf16_instance_part1.cpp`  (+55/-0)
- `library/src/tensor_operation_instance/gpu/grouped_conv3d_fwd_bilinear/wmma/device_grouped_conv3d_fwd_wmma_bilinear_ndhwgc_gkzyxc_ndhwgk_bf16_instance_part2.cpp`  (+55/-0)
- `library/src/tensor_operation_instance/gpu/grouped_conv3d_fwd_bilinear/wmma/device_grouped_conv3d_fwd_wmma_bilinear_ndhwgc_gkzyxc_ndhwgk_bf16_instance_part3.cpp`  (+55/-0)
- `library/src/tensor_operation_instance/gpu/grouped_conv3d_fwd_bilinear/wmma/device_grouped_conv3d_fwd_wmma_bilinear_ndhwgc_gkzyxc_ndhwgk_bf16_instance_part4.cpp`  (+55/-0)
- `library/src/tensor_operation_instance/gpu/grouped_conv3d_fwd_bilinear/wmma/device_grouped_conv3d_fwd_wmma_bilinear_ndhwgc_gkzyxc_ndhwgk_f16_instance_part1.cpp`  (+55/-0)
- `library/src/tensor_operation_instance/gpu/grouped_conv3d_fwd_bilinear/wmma/device_grouped_conv3d_fwd_wmma_bilinear_ndhwgc_gkzyxc_ndhwgk_f16_instance_part2.cpp`  (+55/-0)

## Key added lines (kernel files)

**`library/include/ck/library/tensor_operation_instance/gpu/grouped_conv_fwd/device_grouped_conv_fwd_wmma_cshufflev3_bilinear_instance.hpp`**
```
namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {
```

**`library/include/ck/library/tensor_operation_instance/gpu/grouped_conv_fwd/device_grouped_conv_fwd_wmma_cshufflev3_scale_instance.hpp`**
```
namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {
```

**`library/include/ck/library/tensor_operation_instance/gpu/grouped_convolution_forward_bilinear.hpp`**
```
void add_device_grouped_conv3d_fwd_wmma_cshufflev3_bilinear_ndhwgc_gkzyxc_ndhwgk_bf16_instances_part1(
std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<3,
ck::Tuple<NDHWGK>,
ck::Tuple<BF16>,
```

**`library/include/ck/library/tensor_operation_instance/gpu/grouped_convolution_forward_scale.hpp`**
```
void add_device_grouped_conv3d_fwd_wmma_cshufflev3_scale_ndhwgc_gkzyxc_ndhwgk_bf16_instances_part1(
std::vector<std::unique_ptr<DeviceGroupedConvFwdMultipleABD<3,
ck::Tuple<>,
ck::Tuple<>,
```

**`library/src/tensor_operation_instance/gpu/grouped_conv3d_fwd_bilinear/wmma/device_grouped_conv3d_fwd_wmma_bilinear_ndhwgc_gkzyxc_ndhwgk_bf16_instance_part1.cpp`**
```
namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {
```
