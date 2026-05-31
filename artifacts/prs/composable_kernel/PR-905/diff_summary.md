# Diff summary

- **files changed:** 22
- **lines:** +1169 / -519
- **kernel-ish files:** 22

## Files (by churn)

- `include/ck/tensor_operation/gpu/grid/gridwise_batched_mha_bwd_xdl_cshuffle_qloop_b2t_v1.hpp`  (+186/-76)
- `include/ck/tensor_operation/gpu/grid/gridwise_batched_mha_bwd_xdl_cshuffle_qloop_b2t_light_v1.hpp`  (+185/-76)
- `include/ck/tensor_operation/gpu/grid/gridwise_batched_mha_bwd_xdl_cshuffle_qloop_b2t_light_v2.hpp`  (+178/-69)
- `include/ck/tensor_operation/gpu/grid/gridwise_batched_mha_bwd_xdl_cshuffle_qloop_b2t_v2.hpp`  (+178/-69)
- `example/52_flash_atten_bias/batched_multihead_attention_bias_backward_v2.cpp`  (+58/-32)
- `include/ck/tensor_operation/gpu/device/impl/device_batched_mha_bwd_xdl_cshuffle_qloop_v2.hpp`  (+38/-33)
- `include/ck/tensor_operation/gpu/device/impl/device_batched_mha_bwd_xdl_cshuffle_qloop_light_v1.hpp`  (+33/-35)
- `include/ck/tensor_operation/gpu/device/impl/device_batched_mha_bwd_xdl_cshuffle_qloop_v1.hpp`  (+33/-35)
- `include/ck/tensor_operation/gpu/device/impl/device_batched_mha_bwd_xdl_cshuffle_qloop_light_v2.hpp`  (+33/-32)
- `include/ck/tensor_operation/gpu/device/impl/device_grouped_mha_bwd_xdl_cshuffle_qloop_v2.hpp`  (+35/-11)
- `include/ck/tensor_operation/gpu/device/impl/device_grouped_mha_bwd_xdl_cshuffle_qloop_light_v2.hpp`  (+35/-10)
- `include/ck/tensor_operation/gpu/device/impl/device_grouped_mha_bwd_xdl_cshuffle_qloop_v1.hpp`  (+35/-10)
- `include/ck/tensor_operation/gpu/device/impl/device_grouped_mha_bwd_xdl_cshuffle_qloop_light_v1.hpp`  (+34/-10)
- `example/52_flash_atten_bias/grouped_multihead_attention_bias_backward_v2.cpp`  (+37/-6)
- `include/ck/tensor_operation/operator_transform/transform_contraction_to_gemm.hpp`  (+23/-0)

## Key added lines (kernel files)

**`example/32_batched_gemm_scale_softmax_gemm/batched_multihead_attention_backward_v2.cpp`**
```
nullptr, //  p_acc0_bias;
nullptr, //  p_acc1_bias;
nullptr, //  p_acc0_bias;
nullptr, //  p_acc1_bias;
```

**`example/32_batched_gemm_scale_softmax_gemm/batched_multihead_attention_backward_v3.cpp`**
```
nullptr, // p_acc0_bias;
nullptr, // p_acc1_bias;
nullptr, // p_acc0_bias;
nullptr, // p_acc1_bias;
```

**`example/32_batched_gemm_scale_softmax_gemm/batched_multihead_attention_train_v2.cpp`**
```
nullptr, //  p_acc0_biases;
nullptr, //  p_acc1_biases;
nullptr, // p_acc0_bias;
nullptr, // p_acc1_bias;
```

**`example/52_flash_atten_bias/batched_multihead_attention_bias_backward_v2.cpp`**
```
using U8    = uint8_t;
std::vector<ck::index_t> d0_gs_ms_ns_lengths{G0, G1, M, N};
std::vector<ck::index_t> d0_gs_ms_ns_strides =
Tensor<Acc0BiasDataType> d0_gs_ms_ns(d0_gs_ms_ns_lengths, d0_gs_ms_ns_strides);
```

**`example/52_flash_atten_bias/grouped_multihead_attention_bias_backward_v2.cpp`**
```
std::vector<void*> p_d0grad;
std::vector<Tensor<Acc0BiasDataType>> d0grad_tensors;
std::vector<DeviceMemPtr> d0grad_tensors_device;
num_byte +=
```
