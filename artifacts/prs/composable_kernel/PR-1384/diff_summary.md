# Diff summary

- **files changed:** 19 (diff was byte-capped; summary is partial)
- **lines:** +4933 / -202
- **kernel-ish files:** 17

## Files (by churn)

- `include/ck/tensor_operation/gpu/grid/gridwise_gemm_xdl_cshuffle_v3_multi_d_ab_scale.hpp`  (+1694/-0)
- `include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_v2_ab_scale.hpp`  (+631/-0)
- `include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_v3_ab_scale.hpp`  (+533/-0)
- `include/ck/tensor_operation/gpu/device/impl/device_gemm_multiple_d_xdl_cshuffle_v3_ab_scale.hpp`  (+516/-0)
- `include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_v1_ab_scale.hpp`  (+418/-0)
- `example/65_gemm_multiply_multiply/gemm_multiply_multiply_xdl_fp8_ab_scale.cpp`  (+316/-0)
- `library/include/ck/library/tensor_operation_instance/gpu/gemm_ab_scale.hpp`  (+226/-0)
- `library/include/ck/library/tensor_operation_instance/gpu/gemm_multiply_multiply.hpp`  (+225/-0)
- `include/ck/tensor_operation/gpu/device/impl/device_gemm_multiple_d_xdl_cshuffle_v3.hpp`  (+2/-192)
- `include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_ab_scale_selector.hpp`  (+117/-0)
- `library/include/ck/library/tensor_operation_instance/gpu/gemm_universal.hpp`  (+81/-3)
- `include/ck/tensor_operation/gpu/device/device_gemm_multiple_d_ab_scale.hpp`  (+65/-0)
- `library/src/tensor_operation_instance/gpu/gemm_ab_scale/device_gemm_ab_scale_xdl_f8_f8_bf16/device_gemm_ab_scale_xdl_f8_f8_bf16_mk_nk_mn_128_128_128.hpp`  (+61/-0)
- `include/ck/tensor_operation/gpu/element/element_wise_operation.hpp`  (+26/-1)
- `library/src/tensor_operation_instance/gpu/gemm_ab_scale/CMakeLists.txt`  (+14/-0)

## Key added lines (kernel files)

**`example/01_gemm/gemm_xdl_fp8_v3.cpp`**
```
224, 256,
2, 16, 16, 0,
2, 16, 16, 0,
```

**`example/65_gemm_multiply_multiply/gemm_multiply_multiply_xdl_fp8_ab_scale.cpp`**
```
template <ck::index_t... Is>
using S = ck::Sequence<Is...>;
using BF16 = ck::bhalf_t;
using FP8  = ck::f8_t;
```

**`include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_ab_scale_selector.hpp`**
```
namespace ck {
enum struct BlockGemmPipelineVersion
v1, // Naive
v2, // Mem
```

**`include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_v1_ab_scale.hpp`**
```
namespace ck {
template <BlockGemmPipelineScheduler BlkGemmPipelineVer,
index_t BlockSize,
typename ADataType,
```

**`include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_v2_ab_scale.hpp`**
```
namespace ck {
template <BlockGemmPipelineScheduler BlkGemmPipelineVer,
index_t BlockSize,
typename ADataType,
```
