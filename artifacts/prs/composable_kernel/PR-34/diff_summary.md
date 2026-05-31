# Diff summary

- **files changed:** 24
- **lines:** +1031 / -621
- **kernel-ish files:** 24

## Files (by churn)

- `composable_kernel/src/kernel_wrapper/gridwise_generic_reduction_second_call_threadwise_reduce_all_dims.cpp`  (+222/-0)
- `composable_kernel/src/kernel_wrapper/gridwise_generic_reduction_second_call_warpwise_reduce_all_dims.cpp`  (+221/-0)
- `composable_kernel/src/kernel_wrapper/gridwise_generic_reduction_second_call_blockwise_reduce_all_dims.cpp`  (+205/-0)
- `composable_kernel/src/kernel_wrapper/gridwise_generic_reduction_first_call_warpwise_reduce_all_dims.cpp`  (+22/-69)
- `composable_kernel/src/kernel_wrapper/gridwise_generic_reduction_first_call_threadwise_reduce_all_dims.cpp`  (+22/-68)
- `composable_kernel/src/kernel_wrapper/gridwise_generic_reduction_first_call_multiblock_reduce_all_dims.cpp`  (+21/-68)
- `composable_kernel/src/kernel_wrapper/gridwise_generic_reduction_first_call_blockwise_reduce_all_dims.cpp`  (+21/-67)
- `composable_kernel/include/utility/reduction_enums.hpp`  (+66/-0)
- `composable_kernel/include/utility/reduction_operator.hpp`  (+32/-33)
- `composable_kernel/include/utility/reduction_common.hpp`  (+4/-55)
- `composable_kernel/src/kernel_wrapper/gridwise_generic_reduction_second_call_threadwise_reduce_partial_dims.cpp`  (+13/-32)
- `composable_kernel/src/kernel_wrapper/gridwise_generic_reduction_second_call_warpwise_reduce_partial_dims.cpp`  (+13/-32)
- `composable_kernel/src/kernel_wrapper/gridwise_generic_reduction_second_call_blockwise_reduce_partial_dims.cpp`  (+12/-31)
- `composable_kernel/src/kernel_wrapper/gridwise_generic_reduction_first_call_multiblock_reduce_partial_dims.cpp`  (+14/-27)
- `composable_kernel/src/kernel_wrapper/gridwise_generic_reduction_first_call_threadwise_reduce_partial_dims.cpp`  (+14/-27)

## Key added lines (kernel files)

**`composable_kernel/include/tensor_operation/gridwise_generic_2d_reduction_blockwise.hpp`**
```
const auto zeroVal = opReduce::GetReductionZeroVal();
StaticBuffer<AddressSpaceEnum_t::Vgpr, dstDataType, 1, true> dstValue_buf;
dstValue_buf(I0) = type_convert<dstDataType>{}(accuValue_buf[I0]);
dstValue_buf(I0) += priorDstValue_buf[I0] * beta;
```

**`composable_kernel/include/tensor_operation/gridwise_generic_2d_reduction_direct_threadwise.hpp`**
```
const auto zeroVal = opReduce::GetReductionZeroVal();
StaticBuffer<AddressSpaceEnum_t::Vgpr, dstDataType, 1, true> dstValue_buf;
dstValue_buf(I0) = type_convert<dstDataType>{}(accuValue_buf[I0]);
dstValue_buf(I0) += priorDstValue_buf[I0] * beta;
```

**`composable_kernel/include/tensor_operation/gridwise_generic_2d_reduction_direct_warpwise.hpp`**
```
const auto zeroVal = opReduce::GetReductionZeroVal();
StaticBuffer<AddressSpaceEnum_t::Vgpr, dstDataType, 1, true> dstValue_buf;
dstValue_buf(I0) = type_convert<dstDataType>{}(accuValue_buf[I0]);
dstValue_buf(I0) += priorDstValue_buf(I0) * beta;
```

**`composable_kernel/include/tensor_operation/gridwise_generic_2d_reduction_multiblock.hpp`**
```
const auto zeroVal = opReduce::GetReductionZeroVal();
const auto zeroVal = opReduce::GetReductionZeroVal();
```

**`composable_kernel/include/tensor_operation/reduction_functions_blockwise.hpp`**
```
compType lAccuData            = opReduce::GetReductionZeroVal();
compType lAccuData            = opReduce::GetReductionZeroVal();
```
