# Diff summary

- **files changed:** 33 (diff was byte-capped; summary is partial)
- **lines:** +1786 / -2164
- **kernel-ish files:** 32

## Files (by churn)

- `composable_kernel/include/tensor_operation/xdlops_gemm.hpp`  (+394/-362)
- `composable_kernel/include/tensor_operation/blockwise_gemm_xdlops.hpp`  (+167/-409)
- `composable_kernel/include/utility/amd_buffer_addressing.hpp`  (+271/-55)
- `host/driver_offline/include/device_convolution_forward_implicit_gemm_v4r4r3_xdlops_nhwc_kyxc_nhwk.hpp`  (+0/-302)
- `host/driver_offline/include/device_convolution_forward_implicit_gemm_v4r4_xdlops_nchw_kcyx_nkhw.hpp`  (+0/-280)
- `composable_kernel/include/tensor_operation/gridwise_gemm_xdlops_v2r3.hpp`  (+91/-187)
- `host/driver_offline/include/device_convolution_forward_implicit_gemm_v4r4r2_xdlops_nhwc_kyxc_nhwk.hpp`  (+0/-229)
- `host/driver_offline/include/device_convolution_backward_weight_implicit_gemm_v4r4r2_xdlops_nchw_kcyx_nkhw.hpp`  (+228/-0)
- `composable_kernel/include/utility/dynamic_buffer.hpp`  (+80/-58)
- `composable_kernel/include/problem_transform/transform_backward_weight_convolution_into_gemm_v4r4r2_nchw_kcyx_nkhw.hpp`  (+129/-0)
- `composable_kernel/include/tensor_description/multi_index_transform.hpp`  (+123/-0)
- `host/driver_offline/include/device_convolution_backward_data_implicit_gemm_v4r1_xdlops_nhwc_kyxc_nhwk.hpp`  (+51/-49)
- `host/driver_offline/include/device_convolution_forward_implicit_gemm_v4r4r2_xdlops_nchw_kcyx_nkhw.hpp`  (+62/-31)
- `composable_kernel/include/tensor_operation/threadwise_tensor_slice_transfer.hpp`  (+39/-38)
- `host/driver_offline/include/device_convolution_forward_implicit_gemm_v4r4r4_xdlops_nhwc_kyxc_nhwk.hpp`  (+27/-34)

## Key added lines (kernel files)

**`composable_kernel/include/problem_transform/transform_backward_weight_convolution_into_gemm_v4r4r2_nchw_kcyx_nkhw.hpp`**
```
namespace ck {
template <typename... Wei,
typename... In,
typename... Out,
```

**`composable_kernel/include/tensor_description/multi_index_transform.hpp`**
```
template <typename LowLengths>
struct Merge_v3_division_mod
static constexpr index_t NDimLow = LowLengths::Size();
using LowerIndex = MultiIndex<NDimLow>;
```

**`composable_kernel/include/tensor_description/multi_index_transform_helper.hpp`**
```
return make_merge_transform_v2_magic_division(low_lengths);
return make_merge_transform_v1_carry_check(low_lengths);
template <typename LowLengths>
__host__ __device__ constexpr auto
```

**`composable_kernel/include/tensor_description/tensor_adaptor.hpp`**
```
is_known &= remove_cvref_t<decltype(Transforms{}[i])>::IsKnownAtCompileTime();
```

**`composable_kernel/include/tensor_description/tensor_descriptor.hpp`**
```
is_known &= remove_cvref_t<decltype(Transforms{}[i])>::IsKnownAtCompileTime();
TensorDesc{}, MultiIndex<remove_cvref_t<TensorDesc>::GetNumOfDimension()>{}));
TensorDesc{}, MultiIndex<remove_cvref_t<TensorDesc>::GetNumOfDimension()>{}));
```
