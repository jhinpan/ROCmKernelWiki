# Diff summary

- **files changed:** 24
- **lines:** +496 / -527
- **kernel-ish files:** 24

## Files (by churn)

- `include/ck/tensor_operation/gpu/grid/gridwise_batched_mha_fwd_xdl_cshuffle_v2.hpp`  (+190/-241)
- `include/ck/tensor_operation/gpu/block/blockwise_dropout.hpp`  (+118/-118)
- `include/ck/tensor_operation/gpu/device/impl/device_batched_mha_fwd_xdl_cshuffle_v2.hpp`  (+23/-24)
- `include/ck/tensor_operation/gpu/device/impl/device_grouped_mha_fwd_xdl_cshuffle_v2.hpp`  (+18/-18)
- `example/52_flash_atten_bias/run_grouped_multihead_attention_bias_forward.inc`  (+15/-11)
- `example/32_batched_gemm_scale_softmax_gemm/batched_multihead_attention_backward_v2.cpp`  (+8/-8)
- `example/32_batched_gemm_scale_softmax_gemm/batched_multihead_attention_backward_v3.cpp`  (+8/-8)
- `example/32_batched_gemm_scale_softmax_gemm/batched_multihead_attention_train_v2.cpp`  (+8/-8)
- `example/52_flash_atten_bias/batched_multihead_attention_bias_backward_v2.cpp`  (+8/-8)
- `example/52_flash_atten_bias/run_batched_multihead_attention_bias_forward.inc`  (+9/-6)
- `example/32_batched_gemm_scale_softmax_gemm/grouped_multihead_attention_backward_v2.cpp`  (+7/-7)
- `example/32_batched_gemm_scale_softmax_gemm/grouped_multihead_attention_backward_v3.cpp`  (+7/-7)
- `example/32_batched_gemm_scale_softmax_gemm/grouped_multihead_attention_train_v2.cpp`  (+7/-7)
- `example/52_flash_atten_bias/grouped_multihead_attention_bias_backward_v2.cpp`  (+7/-7)
- `include/ck/tensor_operation/gpu/grid/gridwise_batched_mha_bwd_xdl_cshuffle_qloop_b2t_light_v1.hpp`  (+7/-7)

## Key added lines (kernel files)

**`example/32_batched_gemm_scale_softmax_gemm/batched_multihead_attention_backward_v2.cpp`**
```
ZDataType p_dropout_in_uint8_t,
ref_dropout.MakeArgument(z_g_m_n, p_g_m_n, p_drop_g_m_n, p_dropout_in_uint8_t, rp_dropout);
float p_dropout                = 1 - p_drop;
ZDataType p_dropout_in_uint8_t = ZDataType(std::floor(p_dropout * 255.0));
```

**`example/32_batched_gemm_scale_softmax_gemm/batched_multihead_attention_backward_v3.cpp`**
```
ZDataType p_dropout_in_uint8_t,
ref_dropout.MakeArgument(z_g_m_n, p_g_m_n, p_drop_g_m_n, p_dropout_in_uint8_t, rp_dropout);
float p_dropout                = 1 - p_drop;
ZDataType p_dropout_in_uint8_t = ZDataType(std::floor(p_dropout * 255.0));
```

**`example/32_batched_gemm_scale_softmax_gemm/batched_multihead_attention_train_v2.cpp`**
```
ZDataType p_dropout_in_uint8_t,
ref_dropout.MakeArgument(z_g_m_n, p_g_m_n, p_drop_g_m_n, p_dropout_in_uint8_t, rp_dropout);
float p_dropout                = 1 - p_drop;
ZDataType p_dropout_in_uint8_t = ZDataType(std::floor(p_dropout * 255.0));
```

**`example/32_batched_gemm_scale_softmax_gemm/grouped_multihead_attention_backward_v2.cpp`**
```
ZDataType p_dropout_in_uint8_t,
ref_dropout.MakeArgument(z_g_m_n, p_g_m_n, p_drop_g_m_n, p_dropout_in_uint8_t, rp_dropout);
float p_dropout                = 1 - p_drop;
ZDataType p_dropout_in_uint8_t = ZDataType(std::floor(p_dropout * 255.0));
```

**`example/32_batched_gemm_scale_softmax_gemm/grouped_multihead_attention_backward_v3.cpp`**
```
ZDataType p_dropout_in_uint8_t,
ref_dropout.MakeArgument(z_g_m_n, p_g_m_n, p_drop_g_m_n, p_dropout_in_uint8_t, rp_dropout);
float p_dropout                = 1 - p_drop;
ZDataType p_dropout_in_uint8_t = ZDataType(std::floor(p_dropout * 255.0));
```
