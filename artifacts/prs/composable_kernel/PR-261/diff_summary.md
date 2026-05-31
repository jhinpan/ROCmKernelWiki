# Diff summary

- **files changed:** 23
- **lines:** +1130 / -99
- **kernel-ish files:** 20

## Files (by churn)

- `example/21_gemm_layernorm/gemm_layernorm_xdl_fp16.cpp`  (+378/-0)
- `include/ck/tensor_operation/gpu/device/device_5ary_elementwise.hpp`  (+333/-0)
- `include/ck/tensor_operation/gpu/grid/gridwise_5ary_Elementwise_1d.hpp`  (+251/-0)
- `example/16_gemm_reduce/gemm_reduce_xdl_mean_squaremean_fp16.cpp`  (+44/-22)
- `example/16_gemm_reduce/gemm_reduce_xdl_max_fp16.cpp`  (+34/-18)
- `profiler/include/profile_gemm_reduce_impl.hpp`  (+24/-17)
- `include/ck/tensor_operation/gpu/device/device_batched_gemm_reduce_xdl_cshuffle.hpp`  (+11/-11)
- `include/ck/tensor_operation/gpu/device/device_gemm_reduce_xdl_cshuffle.hpp`  (+9/-9)
- `include/ck/tensor_operation/gpu/element/element_wise_operation.hpp`  (+18/-0)
- `include/ck/tensor_operation/gpu/device/device_gemm_reduce.hpp`  (+4/-4)
- `include/ck/tensor_operation/gpu/grid/gridwise_gemm_reduce_xdl_cshuffle_v1.hpp`  (+4/-4)
- `library/src/tensor_operation_instance/gpu/gemm_reduce/device_gemm_reduce_xdl_cshuffle_f16_f16_f16_f32_f32_km_kn_mn_instance.cpp`  (+3/-2)
- `library/src/tensor_operation_instance/gpu/gemm_reduce/device_gemm_reduce_xdl_cshuffle_f16_f16_f16_f32_f32_km_nk_mn_instance.cpp`  (+3/-2)
- `library/src/tensor_operation_instance/gpu/gemm_reduce/device_gemm_reduce_xdl_cshuffle_f16_f16_f16_f32_f32_mk_kn_mn_instance.cpp`  (+3/-2)
- `library/src/tensor_operation_instance/gpu/gemm_reduce/device_gemm_reduce_xdl_cshuffle_f16_f16_f16_f32_f32_mk_nk_mn_instance.cpp`  (+3/-2)

## Key added lines (kernel files)

**`example/16_gemm_reduce/gemm_reduce_xdl_max_fp16.cpp`**
```
using GemmAccDataType   = F32;
using ReferenceGemmInstance = ck::tensor_operation::host::ReferenceGemm<ADataType,
BDataType,
CDataType,
```

**`example/16_gemm_reduce/gemm_reduce_xdl_mean_squaremean_fp16.cpp`**
```
using GemmAccDataType   = F32;
using UnaryDivElementOp =
ck::tensor_operation::element_wise::UnaryIdentic<ReduceAccDataType, ReduceAccDataType, true>;
using DxsOutElementOp = ck::Tuple<UnaryDivElementOp, UnaryDivElementOp>;
```

**`example/21_gemm_layernorm/gemm_layernorm_xdl_fp16.cpp`**
```
template <ck::index_t... Is>
using S = ck::Sequence<Is...>;
using F16 = ck::half_t;
using F32 = float;
```

**`include/ck/tensor_operation/gpu/device/device_5ary_elementwise.hpp`**
```
namespace ck {
namespace tensor_operation {
namespace device {
template <typename ADataType,
```

**`include/ck/tensor_operation/gpu/device/device_batched_gemm_reduce_xdl_cshuffle.hpp`**
```
typename DxsAccElementwiseOperation,
const DxsAccElementwiseOperation dxs_out_element_op,
typename DxsAccElementwiseOperation,
DxsAccElementwiseOperation>
```
