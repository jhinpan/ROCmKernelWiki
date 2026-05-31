# Diff summary

- **files changed:** 15
- **lines:** +359 / -449
- **kernel-ish files:** 15

## Files (by churn)

- `include/ck/tensor_operation/gpu/device/impl/device_batched_mha_fwd_xdl_cshuffle_v2.hpp`  (+172/-187)
- `include/ck/tensor_operation/gpu/device/impl/device_grouped_mha_fwd_xdl_cshuffle_v2.hpp`  (+84/-114)
- `include/ck/tensor_operation/gpu/grid/gridwise_batched_mha_fwd_xdl_cshuffle_v2.hpp`  (+51/-86)
- `example/52_flash_atten_bias/run_batched_multihead_attention_bias_forward.inc`  (+14/-17)
- `include/ck/tensor_operation/gpu/device/device_batched_gemm_softmax_gemm_permute.hpp`  (+6/-9)
- `example/52_flash_atten_bias/run_grouped_multihead_attention_bias_forward.inc`  (+5/-9)
- `include/ck/tensor_operation/gpu/device/device_grouped_gemm_softmax_gemm_permute.hpp`  (+7/-7)
- `example/32_batched_gemm_scale_softmax_gemm/batched_multihead_attention_train_v2.cpp`  (+6/-6)
- `example/32_batched_gemm_scale_softmax_gemm/batched_multihead_attention_forward_v2.cpp`  (+2/-2)
- `example/32_batched_gemm_scale_softmax_gemm/grouped_multihead_attention_forward_v2.cpp`  (+2/-2)
- `example/32_batched_gemm_scale_softmax_gemm/grouped_multihead_attention_train_v2.cpp`  (+2/-2)
- `example/32_batched_gemm_scale_softmax_gemm/run_batched_multihead_attention_forward.inc`  (+2/-2)
- `example/32_batched_gemm_scale_softmax_gemm/run_grouped_multihead_attention_forward.inc`  (+2/-2)
- `example/52_flash_atten_bias/batched_multihead_attention_bias_forward_v2.cpp`  (+2/-2)
- `example/52_flash_atten_bias/grouped_multihead_attention_bias_forward_v2.cpp`  (+2/-2)

## Key added lines (kernel files)

**`example/32_batched_gemm_scale_softmax_gemm/batched_multihead_attention_forward_v2.cpp`**
```
using Acc0BiasDataType = void;
using Acc1BiasDataType = void;
```

**`example/32_batched_gemm_scale_softmax_gemm/batched_multihead_attention_train_v2.cpp`**
```
using Acc0BiasDataType = void;
using Acc1BiasDataType = void;
nullptr, //  p_acc0_biases;
nullptr, //  p_acc1_biases;
```

**`example/32_batched_gemm_scale_softmax_gemm/grouped_multihead_attention_forward_v2.cpp`**
```
using Acc0BiasDataType = void;
using Acc1BiasDataType = void;
```

**`example/32_batched_gemm_scale_softmax_gemm/grouped_multihead_attention_train_v2.cpp`**
```
using Acc0BiasDataType = void;
using Acc1BiasDataType = void;
```

**`example/32_batched_gemm_scale_softmax_gemm/run_batched_multihead_attention_forward.inc`**
```
nullptr, // std::array<void*, 1> p_acc0_biases;
nullptr, // std::array<void*, 1> p_acc1_biases;
```
