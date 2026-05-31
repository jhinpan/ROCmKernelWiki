# Diff summary

- **files changed:** 18 (diff was byte-capped; summary is partial)
- **lines:** +5346 / -4
- **kernel-ish files:** 18

## Files (by churn)

- `composable_kernel/include/tensor_operation/gridwise_generic_2d_reduction_blockwise.hpp`  (+613/-0)
- `composable_kernel/include/tensor_operation/gridwise_generic_2d_reduction_direct_warpwise.hpp`  (+532/-0)
- `composable_kernel/include/tensor_operation/gridwise_generic_2d_reduction_direct_threadwise.hpp`  (+491/-0)
- `composable_kernel/include/utility/reduction_operator.hpp`  (+420/-0)
- `composable_kernel/include/tensor_operation/gridwise_generic_2d_reduction_multiblock.hpp`  (+376/-0)
- `composable_kernel/include/tensor_operation/reduction_functions_warpwise.hpp`  (+371/-0)
- `composable_kernel/src/kernel_wrapper/gridwise_generic_reduction_first_call_threadwise_reduce_all_dims.cpp`  (+330/-0)
- `composable_kernel/src/kernel_wrapper/gridwise_generic_reduction_first_call_multiblock_reduce_all_dims.cpp`  (+323/-0)
- `composable_kernel/src/kernel_wrapper/gridwise_generic_reduction_first_call_multiblock_reduce_partial_dims.cpp`  (+323/-0)
- `composable_kernel/src/kernel_wrapper/gridwise_generic_reduction_first_call_blockwise_reduce_partial_dims.cpp`  (+318/-0)
- `composable_kernel/src/kernel_wrapper/gridwise_generic_reduction_first_call_blockwise_reduce_all_dims.cpp`  (+317/-0)
- `composable_kernel/src/kernel_wrapper/gridwise_generic_reduction_first_call_threadwise_reduce_partial_dims.cpp`  (+310/-0)
- `composable_kernel/include/tensor_operation/reduction_functions_blockwise.hpp`  (+271/-0)
- `composable_kernel/include/tensor_operation/reduction_functions_threadwise.hpp`  (+141/-0)
- `composable_kernel/include/utility/reduction_common.hpp`  (+104/-0)

## Key added lines (kernel files)

**`composable_kernel/include/tensor_operation/gridwise_generic_2d_reduction_blockwise.hpp`**
```
namespace ck {
template <index_t BlockSize,
typename srcDataType,
typename dstDataType,
```

**`composable_kernel/include/tensor_operation/gridwise_generic_2d_reduction_direct_threadwise.hpp`**
```
namespace ck {
template <index_t BlockSize,
typename srcDataType,
typename dstDataType,
```

**`composable_kernel/include/tensor_operation/gridwise_generic_2d_reduction_direct_warpwise.hpp`**
```
namespace ck {
template <index_t BlockSize,
typename srcDataType,
typename dstDataType,
```

**`composable_kernel/include/tensor_operation/gridwise_generic_2d_reduction_multiblock.hpp`**
```
namespace ck {
template <index_t BlockSize,
typename srcDataType,
typename dstDataType, // not used together with the beta input
```

**`composable_kernel/include/tensor_operation/reduction_functions_blockwise.hpp`**
```
namespace ck {
template <typename buffer2dDescType,
bool blockIsOneRow,
typename opReduce,
```
