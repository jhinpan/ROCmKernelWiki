# Diff summary

- **files changed:** 4 (diff was byte-capped; summary is partial)
- **lines:** +139 / -140
- **kernel-ish files:** 4

## Files (by churn)

- `example/32_batched_gemm_scale_softmax_gemm/batched_multihead_attention_backward_v3.cpp`  (+40/-37)
- `example/32_batched_gemm_scale_softmax_gemm/batched_multihead_attention_backward_v2.cpp`  (+38/-35)
- `example/32_batched_gemm_scale_softmax_gemm/batched_multihead_attention_train_v2.cpp`  (+37/-34)
- `example/32_batched_gemm_scale_softmax_gemm/grouped_multihead_attention_backward_v2.cpp`  (+24/-34)

## Key added lines (kernel files)

**`example/32_batched_gemm_scale_softmax_gemm/batched_multihead_attention_backward_v2.cpp`**
```
ck::tensor_operation::device::DeviceBatchedMultiheadAttentionBackward_Qloop_Xdl_CShuffle_V1<  NumDimG, NumDimM, NumDimN,
ck::tensor_operation::device::DeviceBatchedMultiheadAttentionBackward_Qloop_Xdl_CShuffle_V1<  NumDimG, NumDimM, NumDimN,
ck::tensor_operation::device::DeviceBatchedMultiheadAttentionBackward_Qloop_Xdl_CShuffle_V2<  NumDimG, NumDimM, NumDimN,
```

**`example/32_batched_gemm_scale_softmax_gemm/batched_multihead_attention_backward_v3.cpp`**
```
using Acc0BiasDataType = void;
using Acc1BiasDataType = void;
static constexpr auto GemmSpec = ck::tensor_operation::device::GemmSpecialization::Default;
ck::tensor_operation::device::DeviceBatchedMultiheadAttentionBackward_Qloop_Xdl_CShuffle_Light_V1<  NumDimG, NumDimM, Nu
```

**`example/32_batched_gemm_scale_softmax_gemm/batched_multihead_attention_train_v2.cpp`**
```
ck::tensor_operation::device::DeviceBatchedMultiheadAttentionBackward_Qloop_Xdl_CShuffle_V1<  NumDimG, NumDimM, NumDimN,
ck::tensor_operation::device::DeviceBatchedMultiheadAttentionBackward_Qloop_Xdl_CShuffle_V1<  NumDimG, NumDimM, NumDimN,
ck::tensor_operation::device::DeviceBatchedMultiheadAttentionBackward_Qloop_Xdl_CShuffle_V2<  NumDimG, NumDimM, NumDimN,
```

**`example/32_batched_gemm_scale_softmax_gemm/grouped_multihead_attention_backward_v2.cpp`**
```
ck::tensor_operation::device::DeviceGroupedMultiheadAttentionBackward_Qloop_Xdl_CShuffle_V1<  NumDimG, NumDimM, NumDimN,
ck::tensor_operation::device::DeviceGroupedMultiheadAttentionBackward_Qloop_Xdl_CShuffle_V1<  NumDimG, NumDimM, NumDimN,
```
