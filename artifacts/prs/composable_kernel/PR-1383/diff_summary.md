# Diff summary

- **files changed:** 11
- **lines:** +4289 / -192
- **kernel-ish files:** 10

## Files (by churn)

- `include/ck/tensor_operation/gpu/grid/gridwise_gemm_xdl_cshuffle_v3_multi_d_ab_scale.hpp`  (+1694/-0)
- `include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_v2_ab_scale.hpp`  (+631/-0)
- `include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_v3_ab_scale.hpp`  (+533/-0)
- `include/ck/tensor_operation/gpu/device/impl/device_gemm_multiple_d_xdl_cshuffle_v3_ab_scale.hpp`  (+513/-0)
- `include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_v1_ab_scale.hpp`  (+418/-0)
- `example/65_gemm_multiply_multiply/gemm_multiply_multiply_xdl_fp8_ab_scale.cpp`  (+318/-0)
- `include/ck/tensor_operation/gpu/device/impl/device_gemm_multiple_d_xdl_cshuffle_v3.hpp`  (+1/-191)
- `include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_ab_scale_selector.hpp`  (+117/-0)
- `include/ck/tensor_operation/gpu/device/device_gemm_multiple_d_ab_scale.hpp`  (+62/-0)
- `example/65_gemm_multiply_multiply/CMakeLists.txt`  (+2/-1)
- `example/65_gemm_multiply_multiply/gemm_multiply_multiply_xdl_fp8.cpp`  (+0/-0)

## Key added lines (kernel files)

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

**`include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_v3_ab_scale.hpp`**
```
namespace ck {
template <BlockGemmPipelineScheduler BlkGemmPipelineVer,
index_t BlockSize,
typename ADataType,
```
