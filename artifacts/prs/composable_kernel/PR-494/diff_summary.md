# Diff summary

- **files changed:** 29 (diff was byte-capped; summary is partial)
- **lines:** +2327 / -1838
- **kernel-ish files:** 26

## Files (by churn)

- `include/ck/tensor_operation/gpu/device/impl/device_batched_gemm_softmax_gemm_permute_xdl_cshuffle.hpp`  (+316/-370)
- `include/ck/tensor_operation/gpu/device/device_grouped_gemm_softmax_gemm_permute_xdl_cshuffle.hpp`  (+235/-313)
- `example/32_batched_gemm_scale_softmax_gemm/grouped_gemm_scale_softmax_gemm_permute_xdl_fp16.cpp`  (+28/-313)
- `example/32_batched_gemm_scale_softmax_gemm/run_grouped_gemm_scale_softmax_gemm_permute.inc`  (+319/-0)
- `example/32_batched_gemm_scale_softmax_gemm/batched_gemm_lower_triangle_scale_softmax_gemm_permute_xdl_fp16.cpp`  (+28/-278)
- `example/32_batched_gemm_scale_softmax_gemm/batched_gemm_scale_softmax_gemm_permute_xdl_fp16.cpp`  (+28/-268)
- `include/ck/tensor_operation/operator_transform/transform_contraction_to_gemm.hpp`  (+288/-0)
- `example/32_batched_gemm_scale_softmax_gemm/run_batched_gemm_scale_softmax_gemm_permute.inc`  (+262/-0)
- `client_example/08_fused_attention/fused_attention.cpp`  (+213/-0)
- `include/ck/tensor_operation/gpu/grid/gridwise_batched_gemm_softmax_gemm_xdl_cshuffle_v1.hpp`  (+58/-103)
- `example/32_batched_gemm_scale_softmax_gemm/grouped_gemm_lower_triangle_scale_softmax_gemm_permute_xdl_fp16.cpp`  (+159/-0)
- `library/include/ck/library/tensor_operation_instance/gpu/batched_gemm_softmax_gemm_permute.hpp`  (+129/-0)
- `library/include/ck/library/tensor_operation_instance/gpu/batched_gemm_masking_scale_softmax_gemm_permute.hpp`  (+0/-100)
- `include/ck/tensor_operation/gpu/device/masking_specialization.hpp`  (+82/-0)
- `include/ck/tensor_operation/gpu/device/device_batched_gemm_softmax_gemm_permute.hpp`  (+39/-28)

## Key added lines (kernel files)

**`client_example/08_fused_attention/fused_attention.cpp`**
```
using AElementOp    = ck::tensor_operation::element_wise::PassThrough;
using B0ElementOp   = ck::tensor_operation::element_wise::PassThrough;
using Acc0ElementOp = ck::tensor_operation::element_wise::Scale;
using B1ElementOp   = ck::tensor_operation::element_wise::PassThrough;
```

**`example/32_batched_gemm_scale_softmax_gemm/batched_gemm_lower_triangle_scale_softmax_gemm_permute_xdl_fp16.cpp`**
```
using Acc0BiasDataType = ck::Tuple<>;
using Acc1BiasDataType = ck::Tuple<>;
static constexpr ck::index_t NumDimG = 2;
static constexpr ck::index_t NumDimM = 1;
```

**`example/32_batched_gemm_scale_softmax_gemm/batched_gemm_scale_softmax_gemm_permute_xdl_fp16.cpp`**
```
using Acc0BiasDataType = ck::Tuple<>;
using Acc1BiasDataType = ck::Tuple<>;
static constexpr ck::index_t NumDimG = 2;
static constexpr ck::index_t NumDimM = 1;
```

**`example/32_batched_gemm_scale_softmax_gemm/grouped_gemm_lower_triangle_scale_softmax_gemm_permute_xdl_fp16.cpp`**
```
Gemm + Softmax + Gemm fused operation. Computes C_g_m_o = Softmax(A_g_m_k * B0_g_k_n) * B1_g_n_o
|-----------------|
|-------------------------------------|
template <ck::index_t... Is>
```

**`example/32_batched_gemm_scale_softmax_gemm/grouped_gemm_scale_softmax_gemm_permute_xdl_fp16.cpp`**
```
using Acc0BiasDataType = ck::Tuple<>;
using Acc1BiasDataType = ck::Tuple<>;
static constexpr ck::index_t NumDimG = 2;
static constexpr ck::index_t NumDimM = 1;
```
