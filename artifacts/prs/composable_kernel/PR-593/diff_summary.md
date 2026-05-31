# Diff summary

- **files changed:** 14
- **lines:** +641 / -131
- **kernel-ish files:** 14

## Files (by churn)

- `include/ck/tensor_operation/gpu/device/impl/device_grouped_multihead_attention_forward_xdl_cshuffle.hpp`  (+152/-47)
- `include/ck/tensor_operation/gpu/grid/gridwise_batched_multihead_attention_forward_xdl_cshuffle.hpp`  (+169/-15)
- `example/32_batched_gemm_scale_softmax_gemm/run_grouped_multihead_attention_forward.inc`  (+80/-25)
- `include/ck/tensor_operation/gpu/grid/gridwise_batched_multihead_attention_backward_xdl_cshuffle_v2.hpp`  (+45/-29)
- `include/ck/tensor_operation/gpu/device/impl/device_batched_multihead_attention_forward_xdl_cshuffle.hpp`  (+61/-0)
- `example/32_batched_gemm_scale_softmax_gemm/run_batched_multihead_attention_forward.inc`  (+44/-6)
- `include/ck/utility/data_type.hpp`  (+36/-0)
- `example/32_batched_gemm_scale_softmax_gemm/grouped_multihead_attention_forward_fp16.cpp`  (+12/-3)
- `example/32_batched_gemm_scale_softmax_gemm/grouped_multihead_attention_forward_bf16.cpp`  (+11/-3)
- `library/include/ck/library/reference_tensor_operation/cpu/reference_dropout.hpp`  (+6/-3)
- `example/32_batched_gemm_scale_softmax_gemm/batched_multihead_attention_forward_bf16.cpp`  (+8/-0)
- `example/32_batched_gemm_scale_softmax_gemm/batched_multihead_attention_forward_fp16.cpp`  (+8/-0)
- `include/ck/tensor_operation/gpu/device/device_grouped_gemm_softmax_gemm_permute.hpp`  (+5/-0)
- `include/ck/tensor_operation/gpu/device/device_batched_gemm_softmax_gemm_permute.hpp`  (+4/-0)

## Key added lines (kernel files)

**`example/32_batched_gemm_scale_softmax_gemm/batched_multihead_attention_forward_bf16.cpp`**
```
using U16  = unsigned short;
using ZDataType        = U16;
ZDataType,
using ReferenceDropoutInstance =
```

**`example/32_batched_gemm_scale_softmax_gemm/batched_multihead_attention_forward_fp16.cpp`**
```
using U16 = unsigned short;
using ZDataType        = U16;
ZDataType,
using ReferenceDropoutInstance =
```

**`example/32_batched_gemm_scale_softmax_gemm/grouped_multihead_attention_forward_bf16.cpp`**
```
using U16  = unsigned short;
using ZDataType        = U16;
ZDataType,
128,         // Gemm1NPerBlock
```

**`example/32_batched_gemm_scale_softmax_gemm/grouped_multihead_attention_forward_fp16.cpp`**
```
using U16 = unsigned short;
using ZDataType        = U16;
ZDataType,
128,         // Gemm1NPerBlock
```

**`example/32_batched_gemm_scale_softmax_gemm/run_batched_multihead_attention_forward.inc`**
```
ck::index_t O = 64;
float p_drop                    = 0.1;
float p_dropout                 = 1 - p_drop;
uint16_t p_dropout_in_16bits    = uint16_t(std::floor(p_dropout * 65535.0));
```
