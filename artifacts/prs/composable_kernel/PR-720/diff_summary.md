# Diff summary

- **files changed:** 7 (diff was byte-capped; summary is partial)
- **lines:** +4414 / -283
- **kernel-ish files:** 6

## Files (by churn)

- `include/ck/tensor_operation/gpu/grid/gridwise_batched_multihead_attention_backward_xdl_cshuffle_pt4.hpp`  (+1708/-0)
- `include/ck/tensor_operation/gpu/device/impl/device_batched_multihead_attention_backward_xdl_cshuffle_v4.hpp`  (+1286/-0)
- `example/32_batched_gemm_scale_softmax_gemm/batched_multihead_attention_backward_v4.cpp`  (+1032/-0)
- `example/32_batched_gemm_scale_softmax_gemm/batched_multihead_attention_backward_v3.cpp`  (+158/-129)
- `include/ck/tensor_operation/gpu/device/impl/device_batched_multihead_attention_backward_xdl_cshuffle_v3.hpp`  (+165/-115)
- `include/ck/tensor_operation/gpu/grid/gridwise_batched_multihead_attention_backward_xdl_cshuffle_pt3.hpp`  (+64/-39)
- `example/32_batched_gemm_scale_softmax_gemm/CMakeLists.txt`  (+1/-0)

## Key added lines (kernel files)

**`example/32_batched_gemm_scale_softmax_gemm/batched_multihead_attention_backward_v3.cpp`**
```
using F16   = ck::half_t;
using BF16  = ck::bhalf_t;
using F32   = float;
using U16   = unsigned short;
```

**`example/32_batched_gemm_scale_softmax_gemm/batched_multihead_attention_backward_v4.cpp`**
```
Backprop for Gemm + Softmax + Gemm fused operation, where forward prop is defined as:
Y_g_m_o = Softmax(alpha * Q_g_m_k * K_g_k_n) * V_g_n_o
Computation graph:
K^T                   V
```

**`include/ck/tensor_operation/gpu/device/impl/device_batched_multihead_attention_backward_xdl_cshuffle_v3.hpp`**
```
typename InputDataType,
typename OutputDataType,
bool HasMainKBlockLoop,
bool Deterministic>
```

**`include/ck/tensor_operation/gpu/device/impl/device_batched_multihead_attention_backward_xdl_cshuffle_v4.hpp`**
```
namespace ck {
namespace tensor_operation {
namespace device {
template <typename GridwiseGemm,
```

**`include/ck/tensor_operation/gpu/grid/gridwise_batched_multihead_attention_backward_xdl_cshuffle_pt3.hpp`**
```
template <typename InputDataType,
typename OutputDataType,
typename ZDataType,
bool Deterministic,
```
