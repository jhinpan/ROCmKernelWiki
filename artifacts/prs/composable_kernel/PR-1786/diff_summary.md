# Diff summary

- **files changed:** 11 (diff was byte-capped; summary is partial)
- **lines:** +5485 / -1
- **kernel-ish files:** 10

## Files (by churn)

- `include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_v2_b_scale.hpp`  (+1248/-0)
- `include/ck/tensor_operation/gpu/grid/gridwise_gemm_xdl_cshuffle_v3_b_scale.hpp`  (+1205/-0)
- `include/ck/tensor_operation/gpu/device/impl/device_gemm_xdl_cshuffle_v3_b_scale.hpp`  (+781/-0)
- `include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_v4_b_scale.hpp`  (+686/-0)
- `include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_v3_b_scale.hpp`  (+530/-0)
- `include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_v1_b_scale.hpp`  (+403/-0)
- `example/01_gemm/gemm_xdl_fp16_pk_i4_v3_b_scale.cpp`  (+357/-0)
- `include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_b_scale_selector.hpp`  (+167/-0)
- `include/ck/tensor_operation/gpu/element/unary_element_wise_operation.hpp`  (+70/-1)
- `include/ck/tensor_operation/gpu/device/device_gemm_v2.hpp`  (+37/-0)
- `example/01_gemm/CMakeLists.txt`  (+1/-0)

## Key added lines (kernel files)

**`example/01_gemm/gemm_xdl_fp16_pk_i4_v3_b_scale.cpp`**
```
using ADataType        = ck::half_t;
using BDataType        = ck::pk_i4_t;
using BScaleDataType   = ck::half_t;
using AccDataType      = float;
```

**`include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_b_scale_selector.hpp`**
```
namespace ck {
enum struct BlockGemmPipelineVersion
v1, // Naive
v2, // Mem
```

**`include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_v1_b_scale.hpp`**
```
namespace ck {
template <BlockGemmPipelineScheduler BlkGemmPipelineVer,
index_t BlockSize,
typename ADataType,
```

**`include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_v2_b_scale.hpp`**
```
namespace ck {
template <BlockGemmPipelineScheduler BlkGemmPipelineVer,
index_t BlockSize,
typename ADataType,
```

**`include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_v3_b_scale.hpp`**
```
namespace ck {
template <BlockGemmPipelineScheduler BlkGemmPipelineVer,
index_t BlockSize,
typename ADataType,
```
