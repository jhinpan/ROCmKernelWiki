# Diff summary

- **files changed:** 32 (diff was byte-capped; summary is partial)
- **lines:** +3169 / -174
- **kernel-ish files:** 28

## Files (by churn)

- `include/ck/tensor_operation/gpu/grid/gridwise_gemm_bias_add_reduce_xdl_cshuffle_v1.hpp`  (+988/-0)
- `include/ck/tensor_operation/gpu/device/device_gemm_bias_add_reduce_xdl_cshuffle.hpp`  (+813/-0)
- `example/21_gemm_layernorm/gemm_bias_relu_add_layernorm_xdl_fp16.cpp`  (+424/-0)
- `profiler/include/profile_gemm_bias_add_reduce_impl.hpp`  (+388/-0)
- `library/src/tensor_operation_instance/gpu/gemm_bias_add_reduce/device_gemm_bias_add_reduce_xdl_cshuffle_f16_f16_f16_f32_f32_km_kn_mn_instance.cpp`  (+81/-0)
- `library/src/tensor_operation_instance/gpu/gemm_bias_add_reduce/device_gemm_bias_add_reduce_xdl_cshuffle_f16_f16_f16_f32_f32_km_nk_mn_instance.cpp`  (+81/-0)
- `library/src/tensor_operation_instance/gpu/gemm_bias_add_reduce/device_gemm_bias_add_reduce_xdl_cshuffle_f16_f16_f16_f32_f32_mk_kn_mn_instance.cpp`  (+81/-0)
- `library/src/tensor_operation_instance/gpu/gemm_bias_add_reduce/device_gemm_bias_add_reduce_xdl_cshuffle_f16_f16_f16_f32_f32_mk_nk_mn_instance.cpp`  (+78/-0)
- `include/ck/tensor_operation/gpu/device/device_batched_gemm_reduce_xdl_cshuffle.hpp`  (+34/-32)
- `include/ck/tensor_operation/gpu/device/device_gemm_reduce.hpp`  (+55/-11)
- `include/ck/tensor_operation/gpu/device/device_gemm_reduce_xdl_cshuffle.hpp`  (+28/-27)
- `profiler/include/profile_gemm_reduce_impl.hpp`  (+14/-15)
- `library/src/tensor_operation_instance/gpu/CMakeLists.txt`  (+12/-11)
- `example/16_gemm_reduce/gemm_reduce_xdl_mean_squaremean_fp16.cpp`  (+11/-10)
- `include/ck/tensor_operation/gpu/element/element_wise_operation.hpp`  (+21/-0)

## Key added lines (kernel files)

**`example/16_gemm_reduce/gemm_reduce_xdl_mean_squaremean_fp16.cpp`**
```
using DxsInElementOps  = ck::Tuple<UnaryIdenticElementOp, UnarySquareElementOp>;
using DxsOutElementOps = ck::Tuple<UnaryDivElementOp, UnaryDivElementOp>;
<     Row,     Col,     Row,  F16,   F16,   F16,      F32,      F32,       F32,   DPtrsGlobal,  AElementOp,  BElementOp,
auto dxs_in_element_op  = DxsInElementOps{};
```

**`example/18_batched_gemm_reduce/batched_gemm_reduce_xdl_fp16.cpp`**
```
using DxsInElementOps  = ck::Tuple<UnaryIdenticElementOp, UnarySquareElementOp>;
using DxsOutElementOps = ck::Tuple<UnaryIdenticElementOp, UnaryIdenticElementOp>;
<     Row,     Col,     Row,  F16,   F16,   F16,      F32,      F32,       F32,   DPtrsGlobal,  AElementOp,  BElementOp,
DxsInElementOps{},
```

**`example/21_gemm_layernorm/gemm_bias_relu_add_layernorm_xdl_fp16.cpp`**
```
template <ck::index_t... Is>
using S = ck::Sequence<Is...>;
using F16 = ck::half_t;
using F32 = float;
```

**`example/21_gemm_layernorm/gemm_layernorm_xdl_fp16.cpp`**
```
using DxsInElementOps  = ck::Tuple<UnaryIdenticElementOp, UnarySquareElementOp>;
using DxsOutElementOps = ck::Tuple<UnaryDivElementOp, UnaryDivElementOp>;
<     Row,     Col,     Row,  F16,   F16,   F16,      F32,      F32,       F32,   DPtrsGlobal,  AElementOp,  BElementOp,
auto averageOpInst = UnaryDivElementOp{N};
```

**`include/ck/tensor_operation/gpu/device/device_batched_gemm_reduce_xdl_cshuffle.hpp`**
```
typename DxsReduceAccElementwiseOperation,
const DxsReduceAccElementwiseOperation dxs_out_element_op,
typename DxsReduceAccElementwiseOperation,
struct DeviceBatchedGemmReduce_Xdl_CShuffle
```
