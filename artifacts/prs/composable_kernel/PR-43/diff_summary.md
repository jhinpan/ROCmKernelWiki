# Diff summary

- **files changed:** 19 (diff was byte-capped; summary is partial)
- **lines:** +4428 / -439
- **kernel-ish files:** 19

## Files (by churn)

- `composable_kernel/include/tensor_operation/gridwise_gemm_xdlops_v2r4.hpp`  (+666/-0)
- `composable_kernel/include/tensor_operation/gridwise_generic_2d_reduction_blockwise.hpp`  (+625/-0)
- `composable_kernel/include/tensor_operation/gridwise_generic_2d_reduction_direct_warpwise.hpp`  (+544/-0)
- `composable_kernel/include/tensor_operation/gridwise_generic_2d_reduction_direct_threadwise.hpp`  (+503/-0)
- `composable_kernel/include/tensor_operation/gridwise_gemm_xdlops_v2r3.hpp`  (+182/-260)
- `composable_kernel/include/tensor_operation/gridwise_generic_2d_reduction_multiblock.hpp`  (+376/-0)
- `composable_kernel/include/tensor_operation/reduction_functions_warpwise.hpp`  (+371/-0)
- `composable_kernel/include/tensor_operation/reduction_functions_blockwise.hpp`  (+271/-0)
- `composable_kernel/include/problem_transform/transform_backward_weight_convolution_into_gemm_v4r4r2_atomic_nchw_kcyx_nkhw.hpp`  (+147/-0)
- `composable_kernel/include/problem_transform/transform_backward_weight_convolution_into_gemm_v4r4r4_atomic_nhwc_kyxc_nhwk.hpp`  (+147/-0)
- `composable_kernel/include/problem_transform/transform_backward_weight_convolution_into_gemm_v4r4r5_nhwc_kyxc_nhwk.hpp`  (+144/-0)
- `composable_kernel/include/tensor_operation/reduction_functions_threadwise.hpp`  (+141/-0)
- `composable_kernel/include/problem_transform/transform_backward_weight_convolution_into_gemm_v4r4r4_nhwc_kyxc_nhwk.hpp`  (+132/-0)
- `composable_kernel/include/problem_transform/transform_backward_data_convolution_into_gemm_v4r1r2_nhwc_kyxc_nhwk.hpp`  (+90/-13)
- `composable_kernel/include/tensor_operation/xdlops_gemm.hpp`  (+25/-75)

## Key added lines (kernel files)

**`composable_kernel/include/problem_transform/transform_backward_data_convolution_into_gemm_v4r1r2_nhwc_kyxc_nhwk.hpp`**
```
typename IYTilda,
typename IXTilda,
IYTilda i_ytilda,
IXTilda i_xtilda,
```

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
