# Diff summary

- **files changed:** 15 (diff was byte-capped; summary is partial)
- **lines:** +1414 / -825
- **kernel-ish files:** 15

## Files (by churn)

- `example/52_flash_atten_bias/grouped_multihead_attention_bias_backward_v2.cpp`  (+115/-77)
- `example/52_flash_atten_bias/batched_multihead_attention_bias_backward_v2.cpp`  (+115/-75)
- `example/32_batched_gemm_scale_softmax_gemm/grouped_multihead_attention_train_v2.cpp`  (+112/-74)
- `example/32_batched_gemm_scale_softmax_gemm/batched_multihead_attention_backward_v2.cpp`  (+113/-71)
- `example/32_batched_gemm_scale_softmax_gemm/grouped_multihead_attention_backward_v3.cpp`  (+109/-71)
- `example/32_batched_gemm_scale_softmax_gemm/grouped_multihead_attention_backward_v2.cpp`  (+108/-70)
- `example/32_batched_gemm_scale_softmax_gemm/batched_multihead_attention_backward_v3.cpp`  (+109/-67)
- `example/32_batched_gemm_scale_softmax_gemm/batched_multihead_attention_train_v2.cpp`  (+106/-70)
- `example/52_flash_atten_bias/run_grouped_multihead_attention_bias_forward_v2.inc`  (+77/-63)
- `example/32_batched_gemm_scale_softmax_gemm/run_grouped_multihead_attention_forward.inc`  (+72/-57)
- `example/52_flash_atten_bias/run_batched_multihead_attention_bias_forward_v2.inc`  (+65/-55)
- `example/32_batched_gemm_scale_softmax_gemm/run_batched_multihead_attention_forward.inc`  (+60/-50)
- `include/ck/tensor_operation/gpu/device/impl/device_batched_mha_bwd_xdl_cshuffle_qloop_light_v1.hpp`  (+92/-10)
- `include/ck/tensor_operation/gpu/device/impl/device_batched_mha_bwd_xdl_cshuffle_qloop_light_v2.hpp`  (+91/-8)
- `include/ck/tensor_operation/gpu/device/impl/device_batched_mha_bwd_xdl_cshuffle_qloop_v1.hpp`  (+70/-7)

## Key added lines (kernel files)

**`example/32_batched_gemm_scale_softmax_gemm/batched_multihead_attention_backward_v2.cpp`**
```
ck::index_t M    = 512;
ck::index_t N    = 512;
ck::index_t K    = DIM;
ck::index_t O    = DIM;
```

**`example/32_batched_gemm_scale_softmax_gemm/batched_multihead_attention_backward_v3.cpp`**
```
ck::index_t M    = 512;
ck::index_t N    = 512;
ck::index_t K    = DIM;
ck::index_t O    = DIM;
```

**`example/32_batched_gemm_scale_softmax_gemm/batched_multihead_attention_train_v2.cpp`**
```
ck::index_t N    = 500; // 512
ck::index_t M    = 500; // 512
ck::index_t K    = DIM;
ck::index_t O    = DIM;
```

**`example/32_batched_gemm_scale_softmax_gemm/grouped_multihead_attention_backward_v2.cpp`**
```
int h_ratio  = 1; // G1Q / G1KV
else if(argc == 8)
p_drop  = std::stof(argv[4]);
h_ratio = std::stof(argv[5]);
```

**`example/32_batched_gemm_scale_softmax_gemm/grouped_multihead_attention_backward_v3.cpp`**
```
int h_ratio  = 1; // G1Q / G1KV
else if(argc == 8)
p_drop  = std::stof(argv[4]);
h_ratio = std::stof(argv[5]);
```
