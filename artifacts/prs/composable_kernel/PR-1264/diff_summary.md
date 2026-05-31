# Diff summary

- **files changed:** 25 (diff was byte-capped; summary is partial)
- **lines:** +4047 / -590
- **kernel-ish files:** 22

## Files (by churn)

- `include/ck/tensor_operation/gpu/grid/gridwise_gemm_xdl_cshuffle_v3_multi_abd.hpp`  (+2491/-0)
- `include/ck/tensor_operation/gpu/device/impl/device_gemm_multiple_abd_xdl_cshuffle.hpp`  (+520/-461)
- `example/60_gemm_multi_ABD/gemm_multi_ABD_xdl_multiply_bias_fastgelu_bf16_i8.cpp`  (+274/-0)
- `example/60_gemm_multi_ABD/gemm_multi_ABD_xdl_fastgelu_bf16_i8.cpp`  (+273/-0)
- `client_example/30_gemm_bf16Aint8B/gemm_xdl_multiply_bf16_i8.cpp`  (+220/-0)
- `include/ck/tensor_operation/gpu/element/binary_element_wise_operation.hpp`  (+75/-9)
- `include/ck/tensor_operation/gpu/thread/threadwise_tensor_slice_transfer_v7r2.hpp`  (+57/-16)
- `library/include/ck/library/tensor_operation_instance/device_operation_instance_factory.hpp`  (+20/-17)
- `include/ck/tensor_operation/gpu/element/element_wise_operation.hpp`  (+29/-0)
- `include/ck/tensor_operation/gpu/block/thread_group_tensor_slice_transfer_v7r2.hpp`  (+16/-10)
- `example/60_gemm_multi_ABD/gemm_multi_ABD_xdl_bias_fastgelu_bf16_i8.cpp`  (+14/-11)
- `library/include/ck/library/tensor_operation_instance/gpu/gemm_multi_abd.hpp`  (+12/-12)
- `include/ck/utility/type.hpp`  (+4/-17)
- `client_example/30_gemm_bf16Aint8B/gemm_xdl_bf16_i8.cpp`  (+7/-7)
- `include/ck/tensor_operation/gpu/grid/gridwise_gemm_multiple_abd_xdl_cshuffle.hpp`  (+2/-12)

## Key added lines (kernel files)

**`client_example/30_gemm_bf16Aint8B/gemm_bias_fastgelu_xdl_bf16_i8.cpp`**
```
using B0Layout = Row;
using Multiply    = ck::tensor_operation::element_wise::Multiply;
using BElementOp   = Multiply;
```

**`client_example/30_gemm_bf16Aint8B/gemm_bias_xdl_bf16_i8.cpp`**
```
using A0Layout = Row;
using Multiply    = ck::tensor_operation::element_wise::Multiply;
using BElementOp   = Multiply;
```

**`client_example/30_gemm_bf16Aint8B/gemm_xdl_bf16_i8.cpp`**
```
using B0Layout = Row;
using Multiply    = ck::tensor_operation::element_wise::Multiply;
using BElementOp   = Multiply;
ck::index_t M = 4096;
```

**`client_example/30_gemm_bf16Aint8B/gemm_xdl_gelu_bf16_i8.cpp`**
```
using B0Layout = Row;
using Multiply    = ck::tensor_operation::element_wise::Multiply;
using BElementOp   = Multiply;
```

**`client_example/30_gemm_bf16Aint8B/gemm_xdl_multiply_bf16_i8.cpp`**
```
template <ck::index_t... Is>
using S = ck::Sequence<Is...>;
using BF16 = ck::bhalf_t;
using I8   = int8_t;
```
