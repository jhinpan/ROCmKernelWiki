# Diff summary

- **files changed:** 18
- **lines:** +157 / -337
- **kernel-ish files:** 18

## Files (by churn)

- `host/host_tensor/include/host_gemm.hpp`  (+0/-156)
- `composable_kernel/include/utility/data_type.hpp`  (+29/-36)
- `host/host_tensor/include/host_tensor.hpp`  (+23/-35)
- `host/host_tensor/include/host_tensor_generator.hpp`  (+17/-18)
- `composable_kernel/include/tensor_operation/reduction_functions_blockwise.hpp`  (+17/-17)
- `composable_kernel/include/tensor_operation/gridwise_generic_2d_reduction_blockwise.hpp`  (+10/-12)
- `composable_kernel/include/tensor_operation/gridwise_generic_2d_reduction_direct_threadwise.hpp`  (+10/-12)
- `composable_kernel/include/tensor_operation/gridwise_generic_2d_reduction_direct_warpwise.hpp`  (+10/-12)
- `composable_kernel/include/utility/inner_product.hpp`  (+4/-12)
- `host/driver_offline/src/conv_fwd_driver_offline.cpp`  (+6/-6)
- `host/host_tensor/src/host_tensor.cpp`  (+10/-0)
- `composable_kernel/include/utility/reduction_operator.hpp`  (+4/-4)
- `profiler/include/profile_conv.hpp`  (+4/-4)
- `profiler/include/profile_gemm.hpp`  (+4/-4)
- `composable_kernel/include/tensor_operation/threadwise_tensor_slice_transfer.hpp`  (+3/-3)

## Key added lines (kernel files)

**`composable_kernel/include/tensor_operation/gridwise_generic_2d_reduction_blockwise.hpp`**
```
p_src_global, src2dDesc.GetElementSpaceSize(), type_convert<srcDataType>(zeroVal));
accuValue_buf(I0) *= type_convert<compType>(alpha);
dstValue_buf(I0) = type_convert<dstDataType>(accuValue_buf[I0]);
p_src_global, src2dDesc.GetElementSpaceSize(), type_convert<srcDataType>(zeroVal));
```

**`composable_kernel/include/tensor_operation/gridwise_generic_2d_reduction_direct_threadwise.hpp`**
```
p_src_global, src2dDesc.GetElementSpaceSize(), type_convert<srcDataType>(zeroVal));
accuValue_buf(I0) *= type_convert<compType>(alpha);
dstValue_buf(I0) = type_convert<dstDataType>(accuValue_buf[I0]);
p_src_global, src2dDesc.GetElementSpaceSize(), type_convert<srcDataType>(zeroVal));
```

**`composable_kernel/include/tensor_operation/gridwise_generic_2d_reduction_direct_warpwise.hpp`**
```
p_src_global, src2dDesc.GetElementSpaceSize(), type_convert<srcDataType>(zeroVal));
accuValue_buf(I0) *= type_convert<compType>(alpha);
dstValue_buf(I0) = type_convert<dstDataType>(accuValue_buf[I0]);
p_src_global, src2dDesc.GetElementSpaceSize(), type_convert<srcDataType>(zeroVal));
```

**`composable_kernel/include/tensor_operation/gridwise_generic_2d_reduction_multiblock.hpp`**
```
p_src_global, src2dDesc.GetElementSpaceSize(), type_convert<srcDataType>(zeroVal));
p_src_global, src2dDesc.GetElementSpaceSize(), type_convert<srcDataType>(zeroVal));
```

**`composable_kernel/include/tensor_operation/reduction_functions_blockwise.hpp`**
```
compType opData = type_convert<compType>(block_buffer[offset]);
compType opData1 = type_convert<compType>(block_buffer[offset1]);
compType opData2 = type_convert<compType>(block_buffer[offset2]);
block_buffer(offset1) = type_convert<compType>(opData1);
```
