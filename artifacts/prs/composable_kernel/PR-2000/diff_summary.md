# Diff summary

- **files changed:** 11
- **lines:** +4131 / -137
- **kernel-ish files:** 10

## Files (by churn)

- `include/ck/tensor_operation/gpu/grid/gridwise_gemm_xdl_cshuffle_v3_mx.hpp`  (+2288/-0)
- `include/ck/tensor_operation/gpu/device/impl/device_gemm_xdl_cshuffle_v3_mx.hpp`  (+877/-0)
- `include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_v1_mx.hpp`  (+617/-0)
- `example/67_gemm_microscaling/gemm_mx_common.hpp`  (+165/-113)
- `include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_mx_selector.hpp`  (+69/-0)
- `include/ck/tensor_operation/gpu/device/device_gemm_mx.hpp`  (+50/-0)
- `include/ck/tensor_operation/gpu/thread/threadwise_tensor_slice_transfer.hpp`  (+30/-9)
- `example/67_gemm_microscaling/gemm_mx_fp8.cpp`  (+12/-9)
- `include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_base.hpp`  (+18/-1)
- `include/ck/tensor_operation/gpu/warp/xdlops_gemm.hpp`  (+4/-4)
- `cmake/ClangTidy.cmake`  (+1/-1)

## Key added lines (kernel files)

**`example/67_gemm_microscaling/gemm_mx_common.hpp`**
```
using ck::type_convert;
struct ProblemSizeSplitK final
ck::index_t KBatch = 1;
bool parse_cmd_args(int argc,
```

**`example/67_gemm_microscaling/gemm_mx_fp8.cpp`**
```
using XDataType = ck::half_t;
using CDataType        = ck::half_t;
using CShuffleDataType = CDataType;
using AElementOp = PassThrough; // elementwise transformation for A matrix
```

**`include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_mx_selector.hpp`**
```
namespace ck {
template <BlockGemmPipelineVersion BlkGemmPipelineVer,
BlockGemmPipelineScheduler BlkGemmPipeSche,
index_t ThreadBlockSize,
```

**`include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_v1_mx.hpp`**
```
namespace ck {
template <BlockGemmPipelineScheduler BlkGemmPipelineVer,
index_t ThreadBlockSize,
index_t ScaleBlockSize,
```

**`include/ck/tensor_operation/gpu/device/device_gemm_mx.hpp`**
```
namespace ck {
namespace tensor_operation {
namespace device {
template <typename ALayout,
```
