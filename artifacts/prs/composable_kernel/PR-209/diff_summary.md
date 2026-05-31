# Diff summary

- **files changed:** 23 (diff was byte-capped; summary is partial)
- **lines:** +1108 / -516
- **kernel-ish files:** 20

## Files (by churn)

- `include/ck/tensor_operation/gpu/grid/gridwise_gemm_reduce_xdl_cshuffle_v1.hpp`  (+124/-128)
- `example/16_gemm_reduce/gemm_reduce_xdl_max_fp16.cpp`  (+249/-0)
- `include/ck/tensor_operation/gpu/device/device_batched_gemm_reduce_xdl_cshuffle.hpp`  (+73/-71)
- `include/ck/utility/amd_buffer_addressing.hpp`  (+108/-0)
- `include/ck/utility/generic_memory_space_atomic.hpp`  (+97/-0)
- `example/18_batched_gemm_reduce/batched_gemm_reduce_xdl_fp16.cpp`  (+54/-35)
- `include/ck/tensor_operation/gpu/device/device_gemm_reduce_xdl_cshuffle.hpp`  (+47/-42)
- `example/16_gemm_reduce/gemm_reduce_xdl_sum_squaresum_fp16.cpp`  (+53/-35)
- `library/src/tensor_operation_instance/gpu/batched_gemm_reduce/device_batched_gemm_reduce_xdl_cshuffle_f16_f16_f16_f32_f32_gkm_gkn_gmn_instance.cpp`  (+38/-26)
- `library/src/tensor_operation_instance/gpu/batched_gemm_reduce/device_batched_gemm_reduce_xdl_cshuffle_f16_f16_f16_f32_f32_gkm_gnk_gmn_instance.cpp`  (+38/-26)
- `library/src/tensor_operation_instance/gpu/batched_gemm_reduce/device_batched_gemm_reduce_xdl_cshuffle_f16_f16_f16_f32_f32_gmk_gkn_gmn_instance.cpp`  (+38/-26)
- `library/src/tensor_operation_instance/gpu/batched_gemm_reduce/device_batched_gemm_reduce_xdl_cshuffle_f16_f16_f16_f32_f32_gmk_gnk_gmn_instance.cpp`  (+35/-23)
- `library/src/tensor_operation_instance/gpu/gemm_reduce/device_gemm_reduce_xdl_cshuffle_f16_f16_f16_f32_f32_km_kn_mn_instance.cpp`  (+31/-25)
- `include/ck/tensor_operation/gpu/device/device_gemm_reduce.hpp`  (+29/-22)
- `include/ck/utility/generic_memory_space_atomic_add.hpp`  (+0/-44)

## Key added lines (kernel files)

**`example/16_gemm_reduce/gemm_reduce_xdl_max_fp16.cpp`**
```
template <ck::index_t... Is>
using S = ck::Sequence<Is...>;
using F16 = ck::half_t;
using F32 = float;
```

**`example/16_gemm_reduce/gemm_reduce_xdl_sum_squaresum_fp16.cpp`**
```
using ADataType         = F16;
using BDataType         = F16;
using CDataType         = F16;
using ReduceAccDataType = F32;
```

**`example/18_batched_gemm_reduce/batched_gemm_reduce_xdl_fp16.cpp`**
```
using ADataType         = F16;
using BDataType         = F16;
using CDataType         = F16;
using ReduceAccDataType = F32;
```

**`include/ck/config.hpp`**
```
AtomicMax,
template <InMemoryDataOperationEnum... Is>
struct InMemoryDataOperationEnumSequence
static constexpr int mSize = sizeof...(Is);
```

**`include/ck/tensor_operation/gpu/device/device_batched_gemm_reduce_xdl_cshuffle.hpp`**
```
typename DPtrsGlobal,
typename DxsInElementwiseOperation,
typename DxsOutElementwiseOperation,
DPtrsGlobal p_ds_grid,
```
