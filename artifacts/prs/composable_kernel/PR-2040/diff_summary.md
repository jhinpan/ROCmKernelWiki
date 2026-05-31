# Diff summary

- **files changed:** 14
- **lines:** +1757 / -381
- **kernel-ish files:** 13

## Files (by churn)

- `include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_b_preshuffle_gufusion_dequant_v1.hpp`  (+620/-0)
- `include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_b_preshuffle_gufusion_v1.hpp`  (+573/-0)
- `include/ck/tensor_operation/gpu/grid/gridwise_moe_gemm.hpp`  (+196/-108)
- `include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_b_preshuffle_selector.hpp`  (+99/-42)
- `example/65_gemm_multiply_multiply/moe_gemm1_xdl_pk_i4.cpp`  (+39/-73)
- `example/65_gemm_multiply_multiply/moe_gemm1_xdl_fp8.cpp`  (+55/-50)
- `library/include/ck/library/reference_tensor_operation/cpu/reference_moe_gemm.hpp`  (+74/-15)
- `example/65_gemm_multiply_multiply/moe_gemm2_xdl_fp8.cpp`  (+40/-33)
- `include/ck/tensor_operation/gpu/thread/threadwise_tensor_slice_transfer_v7r3_scatter.hpp`  (+7/-32)
- `include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_base.hpp`  (+22/-3)
- `example/65_gemm_multiply_multiply/moe_gemm2_xdl_pk_i4.cpp`  (+12/-9)
- `include/ck/tensor_operation/gpu/device/impl/device_moe_gemm.hpp`  (+10/-10)
- `include/ck/tensor_operation/gpu/block/thread_group_tensor_slice_transfer_v7r3_scatter.hpp`  (+4/-6)
- `example/65_gemm_multiply_multiply/CMakeLists.txt`  (+6/-0)

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
__host__ __device__ constexpr void operator()<EDataType, EDataType, float, float>(
EDataType& e, const EDataType& c, const float& d0, const float& d1) const
(void)d0;
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
