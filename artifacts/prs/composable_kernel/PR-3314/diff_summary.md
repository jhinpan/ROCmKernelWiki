# Diff summary

- **files changed:** 23 (diff was byte-capped; summary is partial)
- **lines:** +3912 / -548
- **kernel-ish files:** 21

## Files (by churn)

- `include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_wmmaops_v1.hpp`  (+460/-8)
- `include/ck/tensor_operation/gpu/grid/gridwise_gemm_wmma_cshuffle_v3_ab_scale.hpp`  (+308/-101)
- `library/include/ck/library/tensor_operation_instance/gpu/gemm_ab_scale.hpp`  (+374/-20)
- `include/ck/tensor_operation/gpu/device/impl/device_gemm_multiple_d_wmma_cshuffle_v3_ab_scale.hpp`  (+362/-0)
- `include/ck/tensor_operation/gpu/device/impl/device_gemm_multiple_d_wmma_cshuffle_v3_blockscale_bpreshuffle.hpp`  (+360/-0)
- `example/65_gemm_multiply_multiply/gemm_multiply_multiply_wmma_fp8_blockscale_bpreshuffle.cpp`  (+357/-0)
- `include/ck/tensor_operation/gpu/device/device_gemm_multiple_d_ab_scale.hpp`  (+347/-0)
- `example/65_gemm_multiply_multiply/gemm_multiply_multiply_wmma_fp8_ab_scale.cpp`  (+345/-0)
- `include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_wmmaops_v3.hpp`  (+342/-3)
- `include/ck/tensor_operation/gpu/device/impl/device_gemm_multiple_d_wmma_cshuffle_v3_b_preshuffle.hpp`  (+3/-305)
- `include/ck/tensor_operation/gpu/device/impl/device_gemm_wmma_cshuffle_v3_common.hpp`  (+155/-33)
- `include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_wmmaops_base.hpp`  (+115/-35)
- `library/include/ck/library/tensor_operation_instance/gpu/gemm_blockscale_wp.hpp`  (+147/-0)
- `include/ck/tensor_operation/gpu/device/impl/device_gemm_multiple_d_xdl_cshuffle_v3_ab_scale.hpp`  (+89/-13)
- `include/ck/tensor_operation/gpu/grid/gridwise_gemm_xdl_cshuffle_v3_multi_d_ab_scale.hpp`  (+60/-14)

## Key added lines (kernel files)

**`example/65_gemm_multiply_multiply/gemm_multiply_multiply_wmma_fp8_ab_scale.cpp`**
```
template <ck::index_t... Is>
using S = ck::Sequence<Is...>;
using BF16 = ck::bhalf_t;
using FP8  = ck::f8_t;
```

**`example/65_gemm_multiply_multiply/gemm_multiply_multiply_wmma_fp8_blockscale_bpreshuffle.cpp`**
```
template <ck::index_t... Is>
using S = ck::Sequence<Is...>;
using BF16 = ck::bhalf_t;
using FP8  = ck::f8_t;
```

**`include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_wmmaops_base.hpp`**
```
template <index_t ScaleSliceSizeMN,
index_t ScaleSliceStrideMN,
index_t RegSizePerWmma,
typename ThreadDesc>
```

**`include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_wmmaops_v1.hpp`**
```
typename AScaleStruct,
typename BScaleStruct,
typename enable_if<ck::is_same_v<AScaleStruct, Empty>, bool>::type = false>
AScaleStruct&,
```

**`include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_wmmaops_v3.hpp`**
```
using Base::I1;
using Base::I2;
using Base::I3;
b_scale_struct.scale_thread_bufs(
```
