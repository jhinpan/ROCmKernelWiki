# Diff summary

- **files changed:** 16
- **lines:** +267 / -252
- **kernel-ish files:** 16

## Files (by churn)

- `composable_kernel/include/utility/amd_buffer_addressing.hpp`  (+113/-70)
- `composable_kernel/include/utility/dynamic_buffer.hpp`  (+56/-66)
- `composable_kernel/include/tensor_operation/threadwise_tensor_slice_transfer.hpp`  (+25/-34)
- `composable_kernel/include/tensor_operation/threadwise_contraction_dlops.hpp`  (+18/-24)
- `composable_kernel/include/tensor_operation/threadwise_tensor_slice_transfer_v2.hpp`  (+16/-19)
- `composable_kernel/include/tensor_operation/threadwise_gemm_dlops_v3.hpp`  (+9/-12)
- `composable_kernel/include/tensor_operation/blockwise_gemm_dlops_v3.hpp`  (+5/-7)
- `composable_kernel/include/utility/data_type.hpp`  (+11/-0)
- `composable_kernel/src/kernel_wrapper/convolution_forward_implicit_gemm_v6r1_dlops_nchw_kcyx_nkhw.cpp`  (+2/-7)
- `composable_kernel/include/tensor_description/tensor_descriptor.hpp`  (+3/-4)
- `composable_kernel/include/tensor_operation/threadwise_tensor_slice_set.hpp`  (+2/-2)
- `composable_kernel/include/utility/tuple_helper.hpp`  (+1/-3)
- `composable_kernel/include/tensor_description/tensor_adaptor.hpp`  (+1/-2)
- `composable_kernel/include/utility/type.hpp`  (+3/-0)
- `composable_kernel/include/utility/array.hpp`  (+1/-1)

## Key added lines (kernel files)

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

**`composable_kernel/include/tensor_operation/blockwise_gemm_dlops_v3.hpp`**
```
static_assert(
is_same<remove_cvref_t<typename ABlockBuffer::type>, remove_cvref_t<FloatA>>::value &&
is_same<remove_cvref_t<typename BThreadBuffer::type>, remove_cvref_t<FloatB>>::value &&
is_same<remove_cvref_t<typename CThreadBuffer::type>, remove_cvref_t<FloatC>>::value &&
```

**`composable_kernel/include/tensor_operation/threadwise_contraction_dlops.hpp`**
```
static_assert(is_known_at_compile_time<remove_cvref_t<AOriginIdx>>::value &&
is_known_at_compile_time<remove_cvref_t<BOriginIdx>>::value &&
is_known_at_compile_time<remove_cvref_t<COriginIdx>>::value,
"wrong! AOriginIdx, BOriginIdx, COringinIdx should be known at compile-time");
```

**`composable_kernel/include/tensor_operation/threadwise_gemm_dlops_v3.hpp`**
```
static_assert(is_known_at_compile_time<remove_cvref_t<AOriginIdx>>::value &&
is_known_at_compile_time<remove_cvref_t<BOriginIdx>>::value &&
is_known_at_compile_time<remove_cvref_t<COriginIdx>>::value,
"wrong! AOriginIdx, BOriginIdx, COringinIdx should be known at compile-time");
```
