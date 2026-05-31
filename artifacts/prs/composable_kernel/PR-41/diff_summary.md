# Diff summary

- **files changed:** 27
- **lines:** +1832 / -351
- **kernel-ish files:** 25

## Files (by churn)

- `composable_kernel/include/tensor_operation/threadwise_tensor_slice_transfer_v3r2.hpp`  (+802/-0)
- `composable_kernel/include/tensor_description/static_tensor.hpp`  (+265/-0)
- `composable_kernel/include/utility/static_buffer.hpp`  (+92/-94)
- `host/driver_offline/src/gemm_driver_offline.cpp`  (+167/-1)
- `host/host_tensor/include/host_gemm.hpp`  (+0/-157)
- `composable_kernel/include/utility/static_buffer_of_vector_type_v2.hpp`  (+100/-0)
- `script/profile_conv.sh`  (+100/-0)
- `composable_kernel/include/utility/transpose_vectors.hpp`  (+87/-0)
- `composable_kernel/include/utility/is_known_at_compile_time.hpp`  (+49/-0)
- `composable_kernel/include/tensor_operation/blockwise_tensor_slice_transfer.hpp`  (+17/-17)
- `composable_kernel/include/utility/statically_indexed_array.hpp`  (+26/-8)
- `script/profile_gemm.sh`  (+24/-0)
- `composable_kernel/include/utility/tuple_helper.hpp`  (+7/-16)
- `device_operation/include/gemm_common.hpp`  (+0/-22)
- `composable_kernel/include/utility/ignore.hpp`  (+21/-0)

## Key added lines (kernel files)

**`composable_kernel/include/tensor_description/multi_index_transform.hpp`**
```
__host__ __device__ static constexpr void CalculateLowerIndex(LowIdx& idx_low,
const UpIdx& idx_up)
__host__ __device__ constexpr void CalculateLowerIndex(LowIdx& idx_low,
const UpIdx& idx_up) const
```

**`composable_kernel/include/tensor_description/static_tensor.hpp`**
```
namespace ck {
template <AddressSpaceEnum_t AddressSpace,
typename T,
typename TensorDesc,
```

**`composable_kernel/include/tensor_description/tensor_adaptor.hpp`**
```
template <index_t I>
__host__ __device__ constexpr index_t GetTopDimensionLength(Number<I> idim) const
template <index_t I>
__host__ __device__ constexpr index_t GetBottomDimensionLength(Number<I> idim) const
```

**`composable_kernel/include/tensor_operation/blockwise_gemm_xdlops.hpp`**
```
StaticBufferOfVectorTypeV2<AddressSpaceEnum_t::Vgpr,
vector_type<FloatAcc, 16>,
MRepeat * NRepeat,
```

**`composable_kernel/include/tensor_operation/blockwise_tensor_slice_transfer.hpp`**
```
ThreadwiseTensorSliceTransfer_v3r2<ThreadSliceLengths,
DstInMemOp,
SrcDimAccessOrder,
DstDimAccessOrder,
```
