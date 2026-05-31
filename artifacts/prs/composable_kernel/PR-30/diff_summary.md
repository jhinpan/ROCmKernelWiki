# Diff summary

- **files changed:** 15
- **lines:** +2914 / -20
- **kernel-ish files:** 15

## Files (by churn)

- `composable_kernel/include/tensor_operation/gridwise_gemm_xdlops_v2r4.hpp`  (+666/-0)
- `host/driver_offline/include/device_convolution_backward_weight_implicit_gemm_v4r4r5_xdlops_atomic_nhwc_kyxc_nhwk.hpp`  (+458/-0)
- `host/driver_offline/include/device_convolution_backward_weight_implicit_gemm_v4r4r4_xdlops_atomic_nhwc_kyxc_nhwk.hpp`  (+290/-0)
- `host/driver_offline/include/device_convolution_backward_weight_implicit_gemm_v4r4r4_xdlops_nhwc_kyxc_nhwk.hpp`  (+276/-0)
- `host/driver_offline/include/device_convolution_backward_weight_implicit_gemm_v4r4r2_xdlops_atomic_nchw_kcyx_nkhw.hpp`  (+258/-0)
- `host/driver_offline/include/driver_gemm_xdlops_v2r4.hpp`  (+209/-0)
- `host/driver_offline/src/conv_wrw_driver_offline.cpp`  (+161/-7)
- `composable_kernel/include/problem_transform/transform_backward_weight_convolution_into_gemm_v4r4r2_atomic_nchw_kcyx_nkhw.hpp`  (+147/-0)
- `composable_kernel/include/problem_transform/transform_backward_weight_convolution_into_gemm_v4r4r4_atomic_nhwc_kyxc_nhwk.hpp`  (+147/-0)
- `composable_kernel/include/problem_transform/transform_backward_weight_convolution_into_gemm_v4r4r5_nhwc_kyxc_nhwk.hpp`  (+144/-0)
- `composable_kernel/include/problem_transform/transform_backward_weight_convolution_into_gemm_v4r4r4_nhwc_kyxc_nhwk.hpp`  (+132/-0)
- `host/driver_offline/include/device_convolution_backward_weight_implicit_gemm_v4r4r2_xdlops_nchw_kcyx_nkhw.hpp`  (+11/-10)
- `host/host_tensor/include/host_tensor_generator.hpp`  (+11/-0)
- `host/driver_offline/include/device_convolution_backward_data_implicit_gemm_v4r1_xdlops_nhwc_kyxc_nhwk.hpp`  (+3/-2)
- `host/host_tensor/include/device.hpp`  (+1/-1)

## Key added lines (kernel files)

**`composable_kernel/include/problem_transform/transform_backward_weight_convolution_into_gemm_v4r4r2_atomic_nchw_kcyx_nkhw.hpp`**
```
namespace ck {
template <typename... Wei,
typename... In,
typename... Out,
```

**`composable_kernel/include/problem_transform/transform_backward_weight_convolution_into_gemm_v4r4r4_atomic_nhwc_kyxc_nhwk.hpp`**
```
namespace ck {
template <typename... In,
typename... Wei,
typename... Out,
```

**`composable_kernel/include/problem_transform/transform_backward_weight_convolution_into_gemm_v4r4r4_nhwc_kyxc_nhwk.hpp`**
```
namespace ck {
template <typename... In,
typename... Wei,
typename... Out,
```

**`composable_kernel/include/problem_transform/transform_backward_weight_convolution_into_gemm_v4r4r5_nhwc_kyxc_nhwk.hpp`**
```
namespace ck {
template <typename... In,
typename... Wei,
typename... Out,
```

**`composable_kernel/include/tensor_operation/gridwise_gemm_xdlops_v2r4.hpp`**
```
namespace ck {
template <typename GridwiseGemm,
typename FloatAB,
typename FloatC,
```
