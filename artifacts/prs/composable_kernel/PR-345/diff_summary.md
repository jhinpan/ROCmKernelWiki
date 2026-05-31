# Diff summary

- **files changed:** 31
- **lines:** +3957 / -31
- **kernel-ish files:** 24

## Files (by churn)

- `include/ck/tensor_operation/gpu/grid/gridwise_batched_gemm_softmax_gemm_xdl_cshuffle_v1.hpp`  (+1021/-0)
- `include/ck/tensor_operation/gpu/device/device_batched_gemm_softmax_gemm_xdl_cshuffle.hpp`  (+916/-0)
- `example/32_batched_gemm_gemm/batched_gemm_softmax_gemm_xdl_fp16.cpp`  (+392/-0)
- `include/ck/tensor_operation/gpu/block/blockwise_gemm_xdlops.hpp`  (+373/-0)
- `profiler/include/profile_batched_gemm_softmax_gemm_impl.hpp`  (+325/-0)
- `include/ck/tensor_operation/gpu/thread/threadwise_tensor_slice_transfer.hpp`  (+112/-2)
- `include/ck/tensor_operation/gpu/block/blockwise_softmax.hpp`  (+96/-0)
- `library/include/ck/library/tensor_operation_instance/gpu/batched_gemm_softmax_gemm.hpp`  (+93/-0)
- `include/ck/tensor_operation/gpu/device/device_batched_gemm_softmax_gemm.hpp`  (+87/-0)
- `include/ck/tensor_operation/gpu/device/device_batched_gemm_gemm.hpp`  (+86/-0)
- `include/ck/tensor_operation/gpu/block/reduction_functions_blockwise.hpp`  (+72/-0)
- `library/src/tensor_operation_instance/gpu/batched_gemm_softmax_gemm/device_batched_gemm_softmax_gemm_xdl_cshuffle_f16_f16_f16_f16_gmk_gnk_gno_gmo_instance.cpp`  (+68/-0)
- `test/batched_gemm_softmax_gemm/test_batched_gemm_softmax_gemm_util.hpp`  (+68/-0)
- `include/ck/utility/statically_indexed_array_multi_index.hpp`  (+55/-11)
- `include/ck/tensor_operation/gpu/warp/xdlops_gemm.hpp`  (+56/-6)

## Key added lines (kernel files)

**`example/18_batched_gemm_reduce/batched_gemm_reduce_xdl_fp16.cpp`**
```
using ReferenceBatchedGemmInstance =
ck::tensor_operation::host::ReferenceBatchedGemm<ADataType,
BDataType,
CDataType,
```

**`example/24_batched_gemm_e_permute/batched_gemm_e_permute_xdl_fp16.cpp`**
```
using ReferenceBatchedGemmInstance = ck::tensor_operation::host::ReferenceBatchedGemm<ADataType,
BDataType,
EDataType,
AccDataType,
```

**`example/32_batched_gemm_gemm/batched_gemm_softmax_gemm_xdl_fp16.cpp`**
```
Gemm + Gemm fused operation. Computes C_m_o = A_m_k * B0_k_n * B1_n_o
|------------|
|---------------------|
template <ck::index_t... Is>
```

**`include/ck/tensor_description/tensor_descriptor.hpp`**
```
__host__ __device__ constexpr auto GetLengths() const
return generate_sequence_v2([&](auto I) { return GetLength(I); }, Number<ndim_visible_>{});
```

**`include/ck/tensor_operation/gpu/block/blockwise_gemm_xdlops.hpp`**
```
template <index_t MNXdlPerWave, index_t MNWaves, index_t MNPerXdl, typename TileDesc_K0_MN_K1>
__host__ __device__ static constexpr auto
MakeGemmMmaTileDescriptor_MN0_MN1_MN2_K(const TileDesc_K0_MN_K1&)
constexpr index_t K0 = TileDesc_K0_MN_K1{}.GetLength(Number<0>{});
```
