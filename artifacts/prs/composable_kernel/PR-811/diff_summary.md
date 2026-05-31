# Diff summary

- **files changed:** 47
- **lines:** +260 / -141
- **kernel-ish files:** 47

## Files (by churn)

- `include/ck/tensor_operation/gpu/device/masking_specialization.hpp`  (+45/-4)
- `profiler/include/profiler/profile_batched_gemm_softmax_gemm_impl.hpp`  (+17/-16)
- `include/ck/tensor_operation/gpu/device/impl/device_grouped_gemm_softmax_gemm_permute_xdl_cshuffle.hpp`  (+9/-4)
- `include/ck/tensor_operation/gpu/device/impl/device_grouped_mha_bwd_xdl_cshuffle_kloop_v1.hpp`  (+9/-4)
- `include/ck/tensor_operation/gpu/device/impl/device_grouped_mha_bwd_xdl_cshuffle_kloop_v2.hpp`  (+9/-4)
- `include/ck/tensor_operation/gpu/device/impl/device_grouped_mha_bwd_xdl_cshuffle_qloop_v1.hpp`  (+9/-4)
- `include/ck/tensor_operation/gpu/device/impl/device_grouped_mha_bwd_xdl_cshuffle_qloop_v2.hpp`  (+9/-4)
- `include/ck/tensor_operation/gpu/device/impl/device_grouped_mha_fwd_xdl_cshuffle_v1.hpp`  (+9/-4)
- `include/ck/tensor_operation/gpu/device/impl/device_grouped_mha_fwd_xdl_cshuffle_v2.hpp`  (+9/-4)
- `include/ck/tensor_operation/gpu/device/impl/device_batched_gemm_softmax_gemm_permute_xdl_cshuffle.hpp`  (+8/-4)
- `include/ck/tensor_operation/gpu/device/impl/device_batched_mha_bwd_xdl_cshuffle_kloop_v1.hpp`  (+8/-4)
- `include/ck/tensor_operation/gpu/device/impl/device_batched_mha_bwd_xdl_cshuffle_kloop_v2.hpp`  (+8/-4)
- `include/ck/tensor_operation/gpu/device/impl/device_batched_mha_bwd_xdl_cshuffle_qloop_v1.hpp`  (+8/-4)
- `include/ck/tensor_operation/gpu/device/impl/device_batched_mha_bwd_xdl_cshuffle_qloop_v2.hpp`  (+8/-4)
- `include/ck/tensor_operation/gpu/device/impl/device_batched_mha_fwd_xdl_cshuffle_v1.hpp`  (+8/-4)

## Key added lines (kernel files)

**`example/32_batched_gemm_scale_softmax_gemm/batched_gemm_lower_triangle_scale_softmax_gemm_permute_xdl_fp16.cpp`**
```
ck::tensor_operation::device::MaskingSpecialization::MaskUpperTriangleFromTopLeft;
```

**`example/32_batched_gemm_scale_softmax_gemm/batched_multihead_attention_backward_v1.cpp`**
```
ck::tensor_operation::device::MaskingSpecialization::MaskUpperTriangleFromTopLeft;
auto M          = s_g_m_n.GetLengths()[1];
const auto mask = DeviceGemmInstance::C0MatrixMask(M, N);
```

**`example/32_batched_gemm_scale_softmax_gemm/batched_multihead_attention_backward_v2.cpp`**
```
ck::tensor_operation::device::MaskingSpecialization::MaskUpperTriangleFromBottomRight;
auto M          = s_g_m_n.GetLengths()[1];
const auto mask = DeviceGemmInstance::C0MatrixMask(M, N);
ck::index_t M  = 253;
```

**`example/32_batched_gemm_scale_softmax_gemm/batched_multihead_attention_train_v1.cpp`**
```
ck::tensor_operation::device::MaskingSpecialization::MaskUpperTriangleFromTopLeft;
auto M          = s_g_m_n.GetLengths()[1];
const auto mask = DeviceGemmInstanceFWD::C0MatrixMask(M, N);
```

**`example/32_batched_gemm_scale_softmax_gemm/batched_multihead_attention_train_v2.cpp`**
```
ck::tensor_operation::device::MaskingSpecialization::MaskUpperTriangleFromTopLeft;
auto M          = s_g_m_n.GetLengths()[1];
const auto mask = DeviceGemmInstanceFWD::C0MatrixMask(M, N);
```
