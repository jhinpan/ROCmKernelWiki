# Diff summary

- **files changed:** 27
- **lines:** +3679 / -165
- **kernel-ish files:** 23

## Files (by churn)

- `include/ck/tensor_operation/gpu/grid/gridwise_batched_gemm_gemm_wmma_cshuffle_v3.hpp`  (+1127/-0)
- `include/ck/tensor_operation/gpu/device/impl/device_batched_gemm_gemm_wmma_cshuffle_v3.hpp`  (+788/-0)
- `example/31_batched_gemm_gemm/run_batched_gemm_gemm_wmma_cshuffle_v3.inc`  (+304/-0)
- `example/31_batched_gemm_gemm/batched_gemm_gemm_wmma_cshuffle_v3_base.inc`  (+276/-0)
- `test/batched_gemm_gemm/test_batched_gemm_gemm_bf16_wmma_cshuffle_v3.cpp`  (+128/-0)
- `test/batched_gemm_gemm/test_batched_gemm_gemm_fp16_wmma_cshuffle_v3.cpp`  (+128/-0)
- `test/batched_gemm_gemm/test_batched_gemm_gemm_util.hpp`  (+3/-120)
- `test/batched_gemm_gemm/test_batched_gemm_gemm_fp16_xdl.cpp`  (+119/-1)
- `library/include/ck/library/tensor_operation_instance/gpu/batched_gemm_gemm.hpp`  (+109/-3)
- `include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_wmmaops_v3.hpp`  (+91/-16)
- `library/src/tensor_operation_instance/gpu/batched_gemm_gemm/device_batched_gemm_gemm_wmma_cshuffle_v3_bf16_bf16_bf16_bf16_gmk_gnk_gno_gmo_instance.cpp`  (+92/-0)
- `library/src/tensor_operation_instance/gpu/batched_gemm_gemm/device_batched_gemm_gemm_wmma_cshuffle_v3_bf16_bf16_bf16_bf16_gmk_gnk_gon_gmo_instance.cpp`  (+92/-0)
- `library/src/tensor_operation_instance/gpu/batched_gemm_gemm/device_batched_gemm_gemm_wmma_cshuffle_v3_f16_f16_f16_f16_gmk_gnk_gno_gmo_instance.cpp`  (+92/-0)
- `library/src/tensor_operation_instance/gpu/batched_gemm_gemm/device_batched_gemm_gemm_wmma_cshuffle_v3_f16_f16_f16_f16_gmk_gnk_gon_gmo_instance.cpp`  (+92/-0)
- `example/31_batched_gemm_gemm/batched_gemm_gemm_wmma_cshuffle_v3_bf16.cpp`  (+37/-0)

## Key added lines (kernel files)

**`example/31_batched_gemm_gemm/batched_gemm_gemm_wmma_cshuffle_v3_base.inc`**
```
Gemm + Gemm fused operation. Computes C_g_m_n = (A_g_m_k * B0_g_k_l) * B1_g_l_n
|------------------|
|-----------------------------|
static constexpr auto PipeSched   = ck::BlockGemmPipelineScheduler::Interwave;
```

**`example/31_batched_gemm_gemm/batched_gemm_gemm_wmma_cshuffle_v3_bf16.cpp`**
```
using BF16 = ck::bhalf_t;
using F32  = float;
using PassThrough = ck::tensor_operation::element_wise::PassThrough;
using ADataType        = BF16;
```

**`example/31_batched_gemm_gemm/batched_gemm_gemm_wmma_cshuffle_v3_fp16.cpp`**
```
using F16 = ck::half_t;
using F32 = float;
using PassThrough = ck::tensor_operation::element_wise::PassThrough;
using ADataType        = F16;
```

**`example/31_batched_gemm_gemm/batched_gemm_gemm_wmma_cshuffle_v3_fp8.cpp`**
```
using PassThrough = ck::tensor_operation::element_wise::PassThrough;
using ADataType        = ck::f8_t;
using B0DataType       = ck::f8_t;
using B1DataType       = ck::f8_t;
```

**`example/31_batched_gemm_gemm/batched_gemm_gemm_wmma_cshuffle_v3_int8.cpp`**
```
using PassThrough = ck::tensor_operation::element_wise::PassThrough;
using ADataType        = int8_t;
using B0DataType       = int8_t;
using B1DataType       = int8_t;
```
