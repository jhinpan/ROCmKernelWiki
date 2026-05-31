# Diff summary

- **files changed:** 39 (diff was byte-capped; summary is partial)
- **lines:** +3716 / -400
- **kernel-ish files:** 36

## Files (by churn)

- `include/ck/tensor_operation/gpu/device/impl/device_gemm_multiple_d_wmma_cshuffle_v3.hpp`  (+410/-0)
- `example/65_gemm_multiply_multiply/gemm_add_add_wmma_fp16.cpp`  (+267/-0)
- `include/ck/tensor_operation/gpu/grid/gridwise_gemm_wmma_cshuffle_v3_common.hpp`  (+174/-92)
- `include/ck/tensor_operation/gpu/grid/gridwise_gemm_wmma_cshuffle_v3.hpp`  (+138/-72)
- `library/include/ck/library/tensor_operation_instance/gpu/gemm_add_add_fastgelu.hpp`  (+185/-22)
- `library/include/ck/library/tensor_operation_instance/gpu/gemm_add_fastgelu.hpp`  (+177/-16)
- `include/ck/tensor_operation/gpu/grid/gridwise_gemm_wmma_cshuffle_v3_b_scale.hpp`  (+116/-65)
- `library/include/ck/library/tensor_operation_instance/gpu/gemm_add_multiply.hpp`  (+163/-1)
- `library/include/ck/library/tensor_operation_instance/gpu/gemm_add.hpp`  (+141/-14)
- `library/include/ck/library/tensor_operation_instance/gpu/gemm_add_relu.hpp`  (+141/-13)
- `example/69_gemm_add_relu/run_gemm_add_relu_example_wmma.inc`  (+146/-0)
- `example/68_gemm_add/run_gemm_add_example_wmma.inc`  (+145/-0)
- `example/69_gemm_add_relu/run_gemm_add_relu_example_xdl.inc`  (+145/-0)
- `example/68_gemm_add/run_gemm_add_example_xdl.inc`  (+144/-0)
- `example/68_gemm_add/common.hpp`  (+114/-0)

## Key added lines (kernel files)

**`example/65_gemm_multiply_multiply/gemm_add_add_wmma_fp16.cpp`**
```
template <ck::index_t... Is>
using S = ck::Sequence<Is...>;
using F16 = ck::half_t;
using F32 = float;
```

**`example/65_gemm_multiply_multiply/gemm_add_add_xdl_fp16.cpp`**
```
float ave_time = invoker.Run(argument, StreamConfig{nullptr, time_kernel, 0, 20, 50});
std::size_t flop      = std::size_t(2) * M * N * K;
std::size_t num_btype = sizeof(A0DataType) * M * K + sizeof(B0DataType) * K * N +
sizeof(D0DataType) * M * N + sizeof(D1DataType) * M * N +
```

**`example/68_gemm_add/common.hpp`**
```
template <ck::index_t... Is>
using S = ck::Sequence<Is...>;
using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;
```

**`example/68_gemm_add/gemm_add_wmma_bf16.cpp`**
```
using ADataType        = BF16;
using BDataType        = BF16;
using AccDataType      = F32;
using CShuffleDataType = F32;
```

**`example/68_gemm_add/gemm_add_wmma_fp16.cpp`**
```
using ADataType        = F16;
using BDataType        = F16;
using AccDataType      = F32;
using CShuffleDataType = F32;
```
