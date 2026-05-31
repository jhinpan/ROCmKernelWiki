# Diff summary

- **files changed:** 15
- **lines:** +540 / -495
- **kernel-ish files:** 14

## Files (by churn)

- `include/ck/tensor_operation/gpu/device/device_batched_gemm_softmax_gemm_xdl_cshuffle.hpp`  (+71/-240)
- `include/ck/tensor_operation/gpu/device/device_batched_gemm_softmax_gemm_permute_xdl_cshuffle.hpp`  (+73/-200)
- `test/batched_gemm_softmax_gemm/test_batched_gemm_softmax_gemm_fp16.cpp`  (+122/-0)
- `test/batched_gemm_softmax_gemm/test_batched_gemm_softmax_gemm_util.hpp`  (+121/-0)
- `include/ck/tensor_operation/gpu/grid/gridwise_batched_gemm_softmax_gemm_xdl_cshuffle_v1.hpp`  (+59/-10)
- `include/ck/tensor_operation/gpu/block/blockwise_softmax.hpp`  (+32/-13)
- `library/src/tensor_operation_instance/gpu/batched_gemm_softmax_gemm/device_batched_gemm_softmax_gemm_xdl_cshuffle_f16_f16_f16_f16_gmk_gnk_gno_gmo_instance.cpp`  (+15/-2)
- `library/src/tensor_operation_instance/gpu/batched_gemm_gemm/device_batched_gemm_gemm_xdl_cshuffle_f16_f16_f16_f16_gmk_gnk_gno_gmo_instance.cpp`  (+13/-3)
- `profiler/include/profile_batched_gemm_softmax_gemm_impl.hpp`  (+10/-3)
- `include/ck/ck.hpp`  (+11/-0)
- `include/ck/tensor_operation/gpu/grid/gridwise_batched_gemm_gemm_xdl_cshuffle_v1.hpp`  (+1/-9)
- `include/ck/tensor_operation/gpu/device/device_batched_gemm_gemm_xdl_cshuffle.hpp`  (+3/-6)
- `example/32_batched_gemm_scale_softmax_gemm/padded_batched_gemm_scale_softmax_gemm_xdl_fp16.cpp`  (+1/-6)
- `test/batched_gemm_gemm/test_batched_gemm_gemm_fp16.cpp`  (+3/-3)
- `example/32_batched_gemm_scale_softmax_gemm/CMakeLists.txt`  (+5/-0)

## Key added lines (kernel files)

**`example/32_batched_gemm_scale_softmax_gemm/padded_batched_gemm_scale_softmax_gemm_xdl_fp16.cpp`**
```
using Acc0ElementOp = ck::tensor_operation::element_wise::Scale;
```

**`include/ck/tensor_operation/gpu/block/blockwise_softmax.hpp`**
```
typename ThreadSliceDesc_M_K,
bool IgnoreNaN = false>
using ThreadwiseMaxReduce = typename conditional<
IgnoreNaN,
```

**`include/ck/tensor_operation/gpu/device/device_batched_gemm_gemm_xdl_cshuffle.hpp`**
```
block_2_ctile_map_))
arg.block_2_ctile_map_))
arg.block_2_ctile_map_);
```

**`include/ck/tensor_operation/gpu/device/device_batched_gemm_softmax_gemm_permute_xdl_cshuffle.hpp`**
```
static_assert(!matrix_padder.PadK, "KPadding is currently not supported");
const auto a_grid_desc_m_k = matrix_padder.PadADescriptor_M_K(a_grid_desc_mraw_kraw);
const auto M = a_grid_desc_m_k.GetLength(I0);
const auto K = a_grid_desc_m_k.GetLength(I1);
```

**`include/ck/tensor_operation/gpu/device/device_batched_gemm_softmax_gemm_xdl_cshuffle.hpp`**
```
static constexpr auto matrix_padder =
GemmGemmPadder<GemmSpec, index_t, index_t, index_t, index_t>{
MPerBlock, NPerBlock, KPerBlock, Gemm1NPerBlock};
static_assert(!matrix_padder.PadK, "KPadding is currently not supported");
```
