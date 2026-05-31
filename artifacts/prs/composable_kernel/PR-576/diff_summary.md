# Diff summary

- **files changed:** 18
- **lines:** +3787 / -106
- **kernel-ish files:** 14

## Files (by churn)

- `include/ck/tensor_operation/gpu/grid/gridwise_batched_gemm_multiple_d_softmax_gemm_xdl_cshuffle_v1.hpp`  (+1329/-0)
- `example/47_gemm_bias_softmax_gemm_permute/gemm_bias_softmax_gemm_permute.cpp`  (+408/-0)
- `profiler/include/profiler/profile_batched_gemm_bias_softmax_gemm_permute_impl.hpp`  (+395/-0)
- `test/batched_gemm_softmax_gemm_permute/test_batched_gemm_bias_softmax_gemm_permute_util.hpp`  (+380/-0)
- `include/ck/tensor_operation/gpu/device/impl/device_batched_gemm_softmax_gemm_permute_xdl_cshuffle.hpp`  (+178/-101)
- `client_example/08_fused_attention/fused_attention_bias.cpp`  (+226/-0)
- `library/include/ck/library/tensor_operation_instance/gpu/batched_gemm_bias_softmax_gemm_permute.hpp`  (+190/-0)
- `test/batched_gemm_softmax_gemm_permute/test_batched_gemm_bias_softmax_gemm_permute_bf16.cpp`  (+182/-0)
- `test/batched_gemm_softmax_gemm_permute/test_batched_gemm_bias_softmax_gemm_permute_fp16.cpp`  (+182/-0)
- `library/src/tensor_operation_instance/gpu/batched_gemm_softmax_gemm_permute/device_batched_gemm_bias_softmax_gemm_permute_xdl_cshuffle_bf16_bf16_bf16_bf16_gmk_gnk_gno_gmo_instance.cpp`  (+133/-0)
- `library/src/tensor_operation_instance/gpu/batched_gemm_softmax_gemm_permute/device_batched_gemm_bias_softmax_gemm_permute_xdl_cshuffle_f16_f16_f16_f16_gmk_gnk_gno_gmo_instance.cpp`  (+133/-0)
- `include/ck/tensor_operation/gpu/element/binary_element_wise_operation.hpp`  (+32/-0)
- `test/batched_gemm_softmax_gemm_permute/CMakeLists.txt`  (+8/-1)
- `include/ck/tensor_operation/gpu/device/device_batched_gemm_softmax_gemm_permute.hpp`  (+4/-4)
- `client_example/08_fused_attention/CMakeLists.txt`  (+3/-0)

## Key added lines (kernel files)

**`client_example/08_fused_attention/fused_attention_bias.cpp`**
```
using AElementOp    = ck::tensor_operation::element_wise::PassThrough;
using B0ElementOp   = ck::tensor_operation::element_wise::PassThrough;
using Acc0ElementOp = ck::tensor_operation::element_wise::ScaleAdd;
using B1ElementOp   = ck::tensor_operation::element_wise::PassThrough;
```

**`example/47_gemm_bias_softmax_gemm_permute/gemm_bias_softmax_gemm_permute.cpp`**
```
using PassThrough = ck::tensor_operation::element_wise::PassThrough;
using AElementOp    = ck::tensor_operation::element_wise::PassThrough;
using B0ElementOp   = ck::tensor_operation::element_wise::PassThrough;
using C0DEElementOp = ck::tensor_operation::element_wise::ScaleAdd;
```

**`include/ck/tensor_operation/gpu/device/device_batched_gemm_softmax_gemm_permute.hpp`**
```
typename C0DEElementwiseOperation,
typename C1DEElementwiseOperation,
C0DEElementwiseOperation c0de_element_op,
C1DEElementwiseOperation c1de_element_op) = 0;
```

**`include/ck/tensor_operation/gpu/device/impl/device_batched_gemm_softmax_gemm_permute_xdl_cshuffle.hpp`**
```
typename D0sPointer,
typename C0DEElementwiseOperation,
typename C1DEElementwiseOperation,
typename C1GridDescriptor_MBlock_MPerBlock_NBlock_NPerBlock,
```

**`include/ck/tensor_operation/gpu/element/binary_element_wise_operation.hpp`**
```
template <>
__host__ __device__ constexpr void
operator()<float>(float& y, const float& x0, const bhalf_t& x1) const
const float x1_tmp = ck::type_convert<float>(x1);
```
