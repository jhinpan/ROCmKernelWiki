# Diff summary

- **files changed:** 21
- **lines:** +1590 / -93
- **kernel-ish files:** 16

## Files (by churn)

- `example/32_batched_gemm_scale_softmax_gemm/batched_gemm_lower_triangle_scale_softmax_gemm_permute_xdl_fp16.cpp`  (+409/-0)
- `profiler/include/profile_batched_gemm_masking_scale_softmax_gemm_permute_impl.hpp`  (+358/-0)
- `test/batched_gemm_masking_scale_softmax_gemm_permute/test_batched_gemm_masking_scale_softmax_gemm_permute_util.hpp`  (+193/-0)
- `test/batched_gemm_masking_scale_softmax_gemm_permute/test_batched_gemm_masking_scale_softmax_gemm_permute_fp16.cpp`  (+179/-0)
- `include/ck/tensor_operation/gpu/grid/gridwise_batched_gemm_softmax_gemm_xdl_cshuffle_v1.hpp`  (+90/-27)
- `library/include/ck/library/tensor_operation_instance/gpu/batched_gemm_masking_scale_softmax_gemm_permute.hpp`  (+100/-0)
- `library/src/tensor_operation_instance/gpu/batched_gemm_masking_scale_softmax_gemm_permute/device_batched_gemm_masking_scale_softmax_gemm_permute_xdl_cshuffle_f16_f16_f16_f16_gmk_gnk_gno_gmo_instance.cpp`  (+85/-0)
- `library/src/tensor_operation_instance/gpu/batched_gemm_softmax_gemm/device_batched_gemm_softmax_gemm_xdl_cshuffle_f16_f16_f16_f16_gmk_gnk_gno_gmo_instance.cpp`  (+29/-30)
- `include/ck/tensor_operation/gpu/device/device_batched_gemm_softmax_gemm_permute_xdl_cshuffle.hpp`  (+39/-7)
- `include/ck/tensor_operation/gpu/device/device_batched_gemm_softmax_gemm_xdl_cshuffle.hpp`  (+39/-7)
- `include/ck/tensor_operation/gpu/device/device_grouped_gemm_softmax_gemm_permute_xdl_cshuffle.hpp`  (+35/-5)
- `test/batched_gemm_softmax_gemm/test_batched_gemm_softmax_gemm_fp16.cpp`  (+8/-8)
- `example/32_batched_gemm_scale_softmax_gemm/grouped_gemm_scale_softmax_gemm_permute_xdl_fp16.cpp`  (+5/-4)
- `example/32_batched_gemm_scale_softmax_gemm/batched_gemm_scale_softmax_gemm_permute_xdl_fp16.cpp`  (+3/-2)
- `example/32_batched_gemm_scale_softmax_gemm/batched_gemm_scale_softmax_gemm_xdl_fp16.cpp`  (+3/-2)

## Key added lines (kernel files)

**`example/32_batched_gemm_scale_softmax_gemm/batched_gemm_lower_triangle_scale_softmax_gemm_permute_xdl_fp16.cpp`**
```
Gemm + Softmax + Gemm fused operation. Computes C_g_m_o = Softmax(A_g_m_k * B0_g_k_n) * B1_g_n_o
|-----------------|
|-------------------------------------|
template <ck::index_t... Is>
```

**`example/32_batched_gemm_scale_softmax_gemm/batched_gemm_scale_softmax_gemm_permute_xdl_fp16.cpp`**
```
static constexpr auto GemmSpec = ck::tensor_operation::device::GemmSpecialization::MNKOPadding;
8,              // CShuffleBlockTransferScalarPerVector_NPerBlock
false>;         // MaskOutUpperTriangle
```

**`example/32_batched_gemm_scale_softmax_gemm/batched_gemm_scale_softmax_gemm_xdl_fp16.cpp`**
```
static constexpr auto GemmSpec = ck::tensor_operation::device::GemmSpecialization::MNKOPadding;
8,              // CShuffleBlockTransferScalarPerVector_NPerBlock
```

**`example/32_batched_gemm_scale_softmax_gemm/grouped_gemm_scale_softmax_gemm_permute_xdl_fp16.cpp`**
```
static constexpr auto GemmSpec = ck::tensor_operation::device::GemmSpecialization::MNKOPadding;
8,              // CShuffleBlockTransferScalarPerVector_NPerBlock
int K     = 40;
int O     = 40 * (rand() % 2 + 1);
```

**`include/ck/tensor_operation/gpu/device/device_batched_gemm_softmax_gemm_permute_xdl_cshuffle.hpp`**
```
typename C0MatrixMask,
const ComputeBasePtrOfStridedBatch compute_base_ptr_of_batch,
const C0MatrixMask c0_matrix_mask)
block_2_ctile_map,
```
