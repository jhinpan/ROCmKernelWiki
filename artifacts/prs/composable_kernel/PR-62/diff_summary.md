# Diff summary

- **files changed:** 19 (diff was byte-capped; summary is partial)
- **lines:** +4303 / -374
- **kernel-ish files:** 19

## Files (by churn)

- `composable_kernel/include/tensor_operation/gridwise_gemm_xdlops_v3r3.hpp`  (+823/-0)
- `composable_kernel/include/tensor_operation/gridwise_gemm_xdlops_v3r2.hpp`  (+784/-0)
- `composable_kernel/include/tensor_operation/gridwise_gemm_xdlops_v3r1.hpp`  (+744/-0)
- `composable_kernel/include/tensor_operation/gridwise_gemm_xdlops_v2r6.hpp`  (+617/-0)
- `composable_kernel/include/tensor_operation/threadwise_tensor_slice_transfer_v1r5.hpp`  (+341/-0)
- `composable_kernel/include/tensor_operation/gridwise_gemm_xdlops_v2r3.hpp`  (+114/-149)
- `composable_kernel/include/tensor_operation/element_wise_operation.hpp`  (+185/-0)
- `composable_kernel/include/tensor_operation/blockwise_tensor_slice_transfer_v6r3.hpp`  (+182/-0)
- `composable_kernel/include/tensor_operation/gridwise_gemm_xdlops_v2r5.hpp`  (+70/-90)
- `composable_kernel/include/tensor_operation/blockwise_tensor_slice_transfer_v6r2.hpp`  (+157/-0)
- `composable_kernel/include/tensor_operation/gridwise_gemm_xdlops_v2r4.hpp`  (+76/-76)
- `composable_kernel/include/tensor_operation/blockwise_tensor_slice_transfer_v6r1.hpp`  (+133/-0)
- `composable_kernel/include/tensor_operation/blockwise_tensor_slice_transfer_v4r1.hpp`  (+26/-12)
- `composable_kernel/include/tensor_description/static_tensor.hpp`  (+20/-15)
- `composable_kernel/include/tensor_operation/threadwise_tensor_slice_transfer_v1r4.hpp`  (+12/-13)

## Key added lines (kernel files)

**`composable_kernel/include/tensor_description/static_tensor.hpp`**
```
__host__ __device__ constexpr StaticTensor() : invalid_element_scalar_value_{0} {}
: invalid_element_scalar_value_{invalid_element_value}
return zero_scalar_value_;
return invalid_element_scalar_value_;
```

**`composable_kernel/include/tensor_operation/blockwise_tensor_slice_transfer_v4r1.hpp`**
```
typename DstElementwiseOperation,
struct BlockwiseTensorSliceTransfer_v4r1
static constexpr auto thread_slice_lengths = BlockSliceLengths{} / ThreadClusterLengths{};
__device__ constexpr BlockwiseTensorSliceTransfer_v4r1(
```

**`composable_kernel/include/tensor_operation/blockwise_tensor_slice_transfer_v5r1.hpp`**
```
struct BlockwiseTensorSliceTransfer_v5r1
__device__ constexpr BlockwiseTensorSliceTransfer_v5r1(const SrcDesc& src_desc,
ThreadwiseTensorSliceTransfer_v5r1<ThreadSliceLengths,
```

**`composable_kernel/include/tensor_operation/blockwise_tensor_slice_transfer_v6r1.hpp`**
```
namespace ck {
template <index_t BlockSize,
typename ElementwiseOperation,
InMemoryDataOperationEnum_t DstInMemOp,
```

**`composable_kernel/include/tensor_operation/blockwise_tensor_slice_transfer_v6r2.hpp`**
```
namespace ck {
template <index_t BlockSize,
typename ElementwiseOperation,
InMemoryDataOperationEnum_t DstInMemOp,
```
