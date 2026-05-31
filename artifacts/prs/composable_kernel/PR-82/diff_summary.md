# Diff summary

- **files changed:** 15 (diff was byte-capped; summary is partial)
- **lines:** +2493 / -2757
- **kernel-ish files:** 14

## Files (by churn)

- `composable_kernel/include/tensor_operation/gridwise_2d_reduction_blockwise.hpp`  (+925/-0)
- `composable_kernel/include/tensor_operation/gridwise_generic_2d_reduction_blockwise.hpp`  (+0/-623)
- `composable_kernel/include/tensor_operation/gridwise_generic_2d_reduction_direct_warpwise.hpp`  (+0/-542)
- `composable_kernel/include/tensor_operation/gridwise_2d_reduction_multiblock_partial_reduce.hpp`  (+514/-0)
- `composable_kernel/include/tensor_operation/gridwise_generic_2d_reduction_direct_threadwise.hpp`  (+0/-501)
- `composable_kernel/include/tensor_operation/gridwise_2d_reduction_threadwise.hpp`  (+435/-0)
- `composable_kernel/include/tensor_operation/gridwise_generic_2d_reduction_multiblock.hpp`  (+0/-376)
- `composable_kernel/include/tensor_operation/reduction_functions_warpwise.hpp`  (+0/-371)
- `composable_kernel/include/tensor_operation/reduction_functions_blockwise.hpp`  (+116/-202)
- `composable_kernel/include/tensor_operation/gridwise_2d_reduction_multiblock_atomic_add.hpp`  (+268/-0)
- `composable_kernel/include/tensor_operation/element_wise_operation.hpp`  (+155/-0)
- `composable_kernel/include/tensor_operation/reduction_functions_threadwise.hpp`  (+0/-141)
- `composable_kernel/include/tensor_operation/gridwise_set_buffer_value.hpp`  (+79/-0)
- `Dockerfile`  (+1/-1)
- `composable_kernel/include/utility/math_v2.hpp`  (+0/-0)

## Key added lines (kernel files)

**`composable_kernel/include/tensor_operation/element_wise_operation.hpp`**
```
template <typename Y, typename X, bool HasDividing = false>
struct UnaryIdentic;
template <>
struct UnaryIdentic<float, float, false>
```

**`composable_kernel/include/tensor_operation/gridwise_2d_reduction_blockwise.hpp`**
```
namespace ck {
template <typename GridwiseReduction,
bool NeedIndices,
typename InDataType,
```

**`composable_kernel/include/tensor_operation/gridwise_2d_reduction_multiblock_atomic_add.hpp`**
```
namespace ck {
template <typename GridwiseReduction,
typename InDataType,
typename OutDataType,
```

**`composable_kernel/include/tensor_operation/gridwise_2d_reduction_multiblock_partial_reduce.hpp`**
```
namespace ck {
template <typename GridwiseReduction,
bool NeedIndices,
typename InDataType,
```

**`composable_kernel/include/tensor_operation/gridwise_2d_reduction_threadwise.hpp`**
```
namespace ck {
template <typename GridwiseReduction,
bool NeedIndices,
typename InDataType,
```
