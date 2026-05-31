# Diff summary

- **files changed:** 11
- **lines:** +73 / -34
- **kernel-ish files:** 11

## Files (by churn)

- `example/ck_tile/03_gemm/universal_gemm_invoker.hpp`  (+24/-19)
- `include/ck_tile/ops/epilogue/cshuffle_epilogue.hpp`  (+33/-5)
- `test/ck_tile/gemm/test_gemm_pipeline_util.hpp`  (+7/-1)
- `include/ck_tile/ops/gemm/kernel/universal_gemm_kernel.hpp`  (+2/-2)
- `include/ck_tile/ops/gemm/kernel/batched_gemm_kernel.hpp`  (+1/-1)
- `include/ck_tile/ops/gemm/kernel/grouped_gemm_kernel.hpp`  (+1/-1)
- `include/ck_tile/ops/gemm_quant/kernel/gemm_quant_kernel.hpp`  (+1/-1)
- `include/ck_tile/ops/gemm_quant/kernel/grouped_gemm_quant_kernel.hpp`  (+1/-1)
- `include/ck_tile/ops/grouped_convolution/kernel/grouped_convolution_backward_data_kernel.hpp`  (+1/-1)
- `include/ck_tile/ops/grouped_convolution/kernel/grouped_convolution_backward_weight_kernel.hpp`  (+1/-1)
- `include/ck_tile/ops/grouped_convolution/kernel/grouped_convolution_forward_kernel.hpp`  (+1/-1)

## Key added lines (kernel files)

**`example/ck_tile/03_gemm/universal_gemm_invoker.hpp`**
```
using GemmEpilogue = ck_tile::CShuffleEpilogue<ck_tile::CShuffleEpilogueProblem<
ADataType,
BDataType,
DsDataType,
```

**`include/ck_tile/ops/epilogue/cshuffle_epilogue.hpp`**
```
index_t BlockedXDLN_PerWarp_ = 1, // The number of continuous xdl_output per warp
bool DoubleSmemBuffer_       = false>
static constexpr bool DoubleSmemBuffer                 = DoubleSmemBuffer_;
static constexpr bool DoubleSmemBuffer                 = Problem::DoubleSmemBuffer;
```

**`include/ck_tile/ops/gemm/kernel/batched_gemm_kernel.hpp`**
```
__shared__ char smem_ptr1[GemmPipeline::GetSmemSize()];
```

**`include/ck_tile/ops/gemm/kernel/grouped_gemm_kernel.hpp`**
```
__shared__ char smem_ptr_1[GemmPipeline::GetSmemSize()];
```

**`include/ck_tile/ops/gemm/kernel/universal_gemm_kernel.hpp`**
```
__shared__ char smem_ptr_1[GemmPipeline::GetSmemSize()];
__shared__ char smem_ptr_1[GemmPipeline::GetSmemSize()];
```
