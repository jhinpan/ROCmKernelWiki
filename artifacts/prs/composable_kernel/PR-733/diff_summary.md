# Diff summary

- **files changed:** 7
- **lines:** +89 / -82
- **kernel-ish files:** 7

## Files (by churn)

- `include/ck/tensor_operation/gpu/grid/gridwise_batched_gemm_multiple_d_softmax_gemm_xdl_cshuffle_v1.hpp`  (+45/-65)
- `include/ck/tensor_operation/gpu/device/impl/device_batched_gemm_softmax_gemm_permute_xdl_cshuffle.hpp`  (+24/-2)
- `library/src/tensor_operation_instance/gpu/batched_gemm_softmax_gemm_permute/device_batched_gemm_bias_softmax_gemm_permute_xdl_cshuffle_bf16_bf16_bf16_bf16_gmk_gnk_gno_gmo_instance.cpp`  (+8/-6)
- `library/src/tensor_operation_instance/gpu/batched_gemm_softmax_gemm_permute/device_batched_gemm_bias_softmax_gemm_permute_xdl_cshuffle_f16_f16_f16_f16_gmk_gnk_gno_gmo_instance.cpp`  (+6/-4)
- `library/src/tensor_operation_instance/gpu/batched_gemm_softmax_gemm_permute/device_batched_gemm_softmax_gemm_permute_xdl_cshuffle_bf16_bf16_bf16_bf16_gmk_gnk_gno_gmo_instance.cpp`  (+2/-2)
- `library/src/tensor_operation_instance/gpu/batched_gemm_softmax_gemm_permute/device_batched_gemm_softmax_gemm_permute_xdl_cshuffle_f16_f16_f16_f16_gmk_gnk_gno_gmo_instance.cpp`  (+2/-2)
- `example/47_gemm_bias_softmax_gemm_permute/gemm_bias_softmax_gemm_permute.cpp`  (+2/-1)

## Key added lines (kernel files)

**`example/47_gemm_bias_softmax_gemm_permute/gemm_bias_softmax_gemm_permute.cpp`**
```
MaskingSpec,    // MaskingSpecialization
```

**`include/ck/tensor_operation/gpu/device/impl/device_batched_gemm_softmax_gemm_permute_xdl_cshuffle.hpp`**
```
int D0sTransferSrcScalarPerVector = 4,
LoopScheduler LoopSched           = LoopScheduler::Default>
MaskingSpec == MaskingSpecialization::MaskOutUpperTriangle,
D0sTransferSrcScalarPerVector>;
```

**`include/ck/tensor_operation/gpu/grid/gridwise_batched_gemm_multiple_d_softmax_gemm_xdl_cshuffle_v1.hpp`**
```
int D0sTransferSrcScalarPerVector = 4,
PipelineVersion PipelineVer       = PipelineVersion::v1>
m0,   // MRepeat
n0,   // NRepeat
```

**`library/src/tensor_operation_instance/gpu/batched_gemm_softmax_gemm_permute/device_batched_gemm_bias_softmax_gemm_permute_xdl_cshuffle_bf16_bf16_bf16_bf16_gmk_gnk_gno_gmo_instance.cpp`**
```
DeviceBatchedGemmSoftmaxGemmPermute_Xdl_CShuffle<  NumDimG, NumDimM, NumDimN, NumDimK, NumDimO,   BF16,    BF16,    BF16
DeviceBatchedGemmSoftmaxGemmPermute_Xdl_CShuffle<  NumDimG, NumDimM, NumDimN, NumDimK, NumDimO,   BF16,    BF16,    BF16
DeviceBatchedGemmSoftmaxGemmPermute_Xdl_CShuffle<  NumDimG, NumDimM, NumDimN, NumDimK, NumDimO,   BF16,    BF16,    BF16
DeviceBatchedGemmSoftmaxGemmPermute_Xdl_CShuffle<  NumDimG, NumDimM, NumDimN, NumDimK, NumDimO,   BF16,    BF16,    BF16
```

**`library/src/tensor_operation_instance/gpu/batched_gemm_softmax_gemm_permute/device_batched_gemm_bias_softmax_gemm_permute_xdl_cshuffle_f16_f16_f16_f16_gmk_gnk_gno_gmo_instance.cpp`**
```
DeviceBatchedGemmSoftmaxGemmPermute_Xdl_CShuffle<  NumDimG, NumDimM, NumDimN, NumDimK, NumDimO,   F16,    F16,    F16,  
DeviceBatchedGemmSoftmaxGemmPermute_Xdl_CShuffle<  NumDimG, NumDimM, NumDimN, NumDimK, NumDimO,   F16,    F16,    F16,  
```
