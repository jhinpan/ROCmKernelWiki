# Diff summary

- **files changed:** 19
- **lines:** +1971 / -492
- **kernel-ish files:** 18

## Files (by churn)

- `include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_b_preshuffle_gufusion_dequant_v1.hpp`  (+621/-0)
- `include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_b_preshuffle_gufusion_v1.hpp`  (+573/-0)
- `include/ck/tensor_operation/gpu/grid/gridwise_moe_gemm.hpp`  (+319/-125)
- `example/65_gemm_multiply_multiply/moe_gemm1_xdl_fp8.cpp`  (+79/-84)
- `example/65_gemm_multiply_multiply/moe_gemm1_xdl_pk_i4.cpp`  (+65/-93)
- `include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_b_preshuffle_selector.hpp`  (+99/-42)
- `library/include/ck/library/reference_tensor_operation/cpu/reference_moe_gemm.hpp`  (+70/-15)
- `example/65_gemm_multiply_multiply/moe_gemm2_xdl_fp8.cpp`  (+43/-39)
- `include/ck/utility/dynamic_buffer.hpp`  (+38/-22)
- `include/ck/tensor_operation/gpu/thread/threadwise_tensor_slice_transfer_v7r3_scatter.hpp`  (+12/-34)
- `include/ck/tensor_operation/gpu/device/impl/device_moe_gemm.hpp`  (+12/-14)
- `example/65_gemm_multiply_multiply/moe_gemm2_xdl_pk_i4.cpp`  (+9/-10)
- `include/ck/tensor_operation/gpu/block/thread_group_tensor_slice_transfer_v7r3_scatter.hpp`  (+7/-7)
- `include/ck/tensor_operation/gpu/thread/threadwise_tensor_slice_transfer_v3r1_gather.hpp`  (+4/-3)
- `include/ck/utility/tuple_helper.hpp`  (+7/-0)

## Key added lines (kernel files)

**`example/65_gemm_multiply_multiply/moe_gemm1_xdl_fp8.cpp`**
```
using CShuffleDataType = EDataType;
(void)d0;
(void)d1;
e = ck::type_convert<EDataType>(c);
```

**`example/65_gemm_multiply_multiply/moe_gemm1_xdl_pk_i4.cpp`**
```
using CShuffleDataType = F16;
using D2Layout = ELayout;
using DsLayout = ck::Tuple<D0Layout, D1Layout, D2Layout>;
__host__ __device__ constexpr void operator()<EDataType, EDataType, float, float>(
```

**`example/65_gemm_multiply_multiply/moe_gemm2_xdl_fp8.cpp`**
```
using CShuffleDataType = F16;
(void)d1;
(void)d2;
e = ck::type_convert<EDataType>(c);
```

**`example/65_gemm_multiply_multiply/moe_gemm2_xdl_pk_i4.cpp`**
```
(void)d1;
(void)d2;
e = ck::type_convert<EDataType>(c * 16);
e = ck::type_convert<EDataType>(c);
```

**`include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_b_preshuffle_gufusion_dequant_v1.hpp`**
```
namespace ck {
template <BlockGemmPipelineScheduler BlkGemmPipelineVer,
index_t BlockSize,
typename ADataType,
```
