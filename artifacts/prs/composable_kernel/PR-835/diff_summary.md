# Diff summary

- **files changed:** 12
- **lines:** +1995 / -87
- **kernel-ish files:** 11

## Files (by churn)

- `example/52_flash_atten_bias/run_grouped_multihead_attention_bias_forward.inc`  (+497/-0)
- `example/52_flash_atten_bias/run_batched_multihead_attention_bias_forward.inc`  (+406/-0)
- `example/52_flash_atten_bias/batched_multihead_attention_bias_forward_v2.cpp`  (+330/-0)
- `example/52_flash_atten_bias/grouped_multihead_attention_bias_forward_v2.cpp`  (+330/-0)
- `include/ck/tensor_operation/gpu/device/impl/device_batched_mha_fwd_xdl_cshuffle_v2.hpp`  (+121/-24)
- `include/ck/tensor_operation/gpu/grid/gridwise_batched_mha_fwd_xdl_cshuffle_v2.hpp`  (+142/-3)
- `include/ck/tensor_operation/gpu/device/impl/device_grouped_mha_fwd_xdl_cshuffle_v2.hpp`  (+119/-24)
- `example/32_batched_gemm_scale_softmax_gemm/batched_multihead_attention_train_v2.cpp`  (+15/-15)
- `example/32_batched_gemm_scale_softmax_gemm/grouped_multihead_attention_train_v2.cpp`  (+15/-15)
- `example/32_batched_gemm_scale_softmax_gemm/batched_multihead_attention_forward_v2.cpp`  (+9/-3)
- `example/32_batched_gemm_scale_softmax_gemm/grouped_multihead_attention_forward_v2.cpp`  (+9/-3)
- `example/52_flash_atten_bias/CMakeLists.txt`  (+2/-0)

## Key added lines (kernel files)

**`example/32_batched_gemm_scale_softmax_gemm/batched_multihead_attention_forward_v2.cpp`**
```
MaskingSpec, // MaskingSpecialization
MaskingSpec, // MaskingSpecialization
MaskingSpec, // MaskingSpecialization
```

**`example/32_batched_gemm_scale_softmax_gemm/batched_multihead_attention_train_v2.cpp`**
```
ck::tensor_operation::device::DeviceBatchedMultiheadAttentionForward_Xdl_CShuffle_V2<  NumDimG, NumDimM, NumDimN, NumDim
ck::tensor_operation::device::DeviceBatchedMultiheadAttentionForward_Xdl_CShuffle_V2<  NumDimG, NumDimM, NumDimN, NumDim
ck::tensor_operation::device::DeviceBatchedMultiheadAttentionForward_Xdl_CShuffle_V2<  NumDimG, NumDimM, NumDimN, NumDim
```

**`example/32_batched_gemm_scale_softmax_gemm/grouped_multihead_attention_forward_v2.cpp`**
```
MaskingSpec, // MaskingSpecialization
MaskingSpec, // MaskingSpecialization
MaskingSpec, // MaskingSpecialization
```

**`example/32_batched_gemm_scale_softmax_gemm/grouped_multihead_attention_train_v2.cpp`**
```
ck::tensor_operation::device::DeviceGroupedMultiheadAttentionForward_Xdl_CShuffle_V2<  NumDimG, NumDimM, NumDimN, NumDim
ck::tensor_operation::device::DeviceGroupedMultiheadAttentionForward_Xdl_CShuffle_V2<  NumDimG, NumDimM, NumDimN, NumDim
ck::tensor_operation::device::DeviceGroupedMultiheadAttentionForward_Xdl_CShuffle_V2<  NumDimG, NumDimM, NumDimN, NumDim
```

**`example/52_flash_atten_bias/batched_multihead_attention_bias_forward_v2.cpp`**
```
Gemm + Softmax + Gemm fused operation. Computes C_g_m_o = Softmax(A_g_m_k * B0_g_k_n) * B1_g_n_o
|-----------------|
|-------------------------------------|
template <ck::index_t... Is>
```
