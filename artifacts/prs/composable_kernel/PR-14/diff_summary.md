# Diff summary

- **files changed:** 14
- **lines:** +63 / -69
- **kernel-ish files:** 14

## Files (by churn)

- `composable_kernel/include/tensor_description/multi_index_transform.hpp`  (+14/-14)
- `composable_kernel/include/tensor_operation/threadwise_tensor_slice_transfer_v2.hpp`  (+9/-9)
- `composable_kernel/include/tensor_operation/gridwise_contraction_dlops_v1r2.hpp`  (+8/-8)
- `composable_kernel/include/tensor_operation/gridwise_gemm_dlops_v1r3.hpp`  (+8/-8)
- `composable_kernel/include/tensor_description/tensor_descriptor_helper.hpp`  (+6/-6)
- `composable_kernel/include/tensor_operation/gridwise_gemm_dlops_v1r2.hpp`  (+6/-6)
- `composable_kernel/include/tensor_operation/gridwise_gemm_xdlops_v2r3.hpp`  (+4/-4)
- `composable_kernel/include/tensor_operation/gridwise_gemm_dlops_v2.hpp`  (+3/-3)
- `composable_kernel/include/utility/math.hpp`  (+0/-6)
- `composable_kernel/include/tensor_description/cluster_descriptor.hpp`  (+1/-1)
- `composable_kernel/include/tensor_description/tensor_adaptor.hpp`  (+1/-1)
- `composable_kernel/include/tensor_description/tensor_descriptor.hpp`  (+1/-1)
- `composable_kernel/include/tensor_operation/blockwise_tensor_slice_transfer.hpp`  (+1/-1)
- `composable_kernel/include/tensor_operation/blockwise_tensor_slice_transfer_v2.hpp`  (+1/-1)

## Key added lines (kernel files)

**`composable_kernel/include/tensor_description/cluster_descriptor.hpp`**
```
__host__ __device__ constexpr auto make_cluster_descriptor(
```

**`composable_kernel/include/tensor_description/multi_index_transform.hpp`**
```
using LowLengthsScan =
decltype(container_reverse_exclusive_scan(LowLengths{}, math::multiplies{}, Number<1>{}));
decltype(make_tuple(container_reduce(LowLengths{}, math::multiplies{}, Number<1>{})));
container_reverse_exclusive_scan(low_lengths, math::multiplies{}, Number<1>{})},
```

**`composable_kernel/include/tensor_description/tensor_adaptor.hpp`**
```
return container_reduce(lengths, math::multiplies{}, Number<1>{});
```

**`composable_kernel/include/tensor_description/tensor_descriptor.hpp`**
```
return container_reduce(lengths, math::multiplies{}, Number<1>{});
```

**`composable_kernel/include/tensor_description/tensor_descriptor_helper.hpp`**
```
__host__ __device__ constexpr auto make_naive_tensor_descriptor(const Tuple<Lengths...>& lengths,
const Tuple<Strides...>& strides)
const auto element_space_size = container_reduce(lengths, math::multiplies{}, Number<1>{});
make_naive_tensor_descriptor_aligned(const Tuple<Lengths...>& lengths, Align align)
```
