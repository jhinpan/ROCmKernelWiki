# Diff summary

- **files changed:** 21 (diff was byte-capped; summary is partial)
- **lines:** +5384 / -14
- **kernel-ish files:** 19

## Files (by churn)

- `include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_v2.hpp`  (+1154/-0)
- `include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_v1.hpp`  (+732/-0)
- `include/ck/tensor_operation/gpu/device/impl/device_gemm_xdl_cshuffle_v3.hpp`  (+687/-0)
- `include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_v5.hpp`  (+667/-0)
- `include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_v4.hpp`  (+597/-0)
- `include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_v3.hpp`  (+439/-0)
- `include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_base.hpp`  (+354/-0)
- `example/01_gemm/run_gemm_example_v2.inc`  (+211/-0)
- `include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_selector.hpp`  (+167/-0)
- `include/ck/tensor_description/multi_index_transform.hpp`  (+85/-0)
- `example/01_gemm/common.hpp`  (+62/-0)
- `example/01_gemm/gemm_xdl_fp16_fp8_v3.cpp`  (+53/-0)
- `example/01_gemm/gemm_xdl_fp16_v3.cpp`  (+48/-0)
- `example/01_gemm/gemm_xdl_fp8_v3.cpp`  (+48/-0)
- `include/ck/tensor_operation/gpu/device/device_gemm_v2.hpp`  (+43/-0)

## Key added lines (kernel files)

**`example/01_gemm/common.hpp`**
```
struct ProblemSizeSplitK final
ck::index_t M = 3840;
ck::index_t N = 4096;
ck::index_t K = 4096;
```

**`example/01_gemm/gemm_xdl_fp16_fp8_v3.cpp`**
```
using ADataType        = ck::f8_t;
using BDataType        = ck::half_t;
using AccDataType      = float;
using CShuffleDataType = ck::half_t;
```

**`example/01_gemm/gemm_xdl_fp16_v3.cpp`**
```
using ADataType        = ck::half_t;
using BDataType        = ck::half_t;
using AccDataType      = float;
using CShuffleDataType = ck::half_t;
```

**`example/01_gemm/gemm_xdl_fp8_v3.cpp`**
```
using ADataType        = ck::f8_t;
using BDataType        = ck::f8_t;
using AccDataType      = float;
using CShuffleDataType = ck::half_t;
```

**`example/01_gemm/run_gemm_example_v2.inc`**
```
template <typename ProblemType>
bool run_gemm(const ProblemType& problem_size, const ExecutionConfig& config)
static_assert(sizeof(ck::int4_t) == sizeof(int8_t));
using namespace ck::literals;
```
