# Diff summary

- **files changed:** 20
- **lines:** +950 / -157
- **kernel-ish files:** 19

## Files (by churn)

- `include/ck/tensor_operation/gpu/grid/gridwise_ab_transfer_wave_tiles_interleave.hpp`  (+275/-0)
- `include/ck/tensor_operation/gpu/device/impl/device_grouped_conv_fwd_multiple_abd_wmma_cshuffle_v3.hpp`  (+78/-83)
- `include/ck/tensor_operation/gpu/grid/gridwise_gemm_wmma_cshuffle_v3_common.hpp`  (+118/-39)
- `include/ck/tensor_operation/gpu/grid/epilogue_direct_store.hpp`  (+145/-0)
- `include/ck/tensor_operation/gpu/grid/gridwise_ab_transfer_wave_tiles.hpp`  (+69/-16)
- `library/include/ck/library/tensor_operation_instance/gpu/grouped_conv_fwd/device_grouped_conv_fwd_wmma_cshufflev3_wave_transfer_instance.hpp`  (+76/-0)
- `library/src/tensor_operation_instance/gpu/grouped_conv2d_fwd/wmma/device_grouped_conv2d_fwd_wmma_cshufflev3_wave_transfer_nhwgc_gkyxc_nhwgk_bf16_instance.cpp`  (+51/-0)
- `library/src/tensor_operation_instance/gpu/grouped_conv2d_fwd/wmma/device_grouped_conv2d_fwd_wmma_cshufflev3_wave_transfer_nhwgc_gkyxc_nhwgk_f16_instance.cpp`  (+51/-0)
- `library/include/ck/library/tensor_operation_instance/gpu/grouped_convolution_forward_wmma_cshufflev3.inc`  (+28/-0)
- `include/ck/tensor_operation/gpu/device/impl/device_batched_gemm_wmma_cshuffle_v3.hpp`  (+9/-3)
- `include/ck/tensor_operation/gpu/device/impl/device_batched_gemm_wmma_cshuffle_v3_b_scale.hpp`  (+9/-3)
- `include/ck/tensor_operation/gpu/device/impl/device_grouped_gemm_wmma_splitk_cshuffle_v3.hpp`  (+8/-4)
- `include/ck/tensor_operation/gpu/device/impl/device_grouped_gemm_multiple_d_wmma_cshuffle_tile_loop_v3.hpp`  (+7/-3)
- `include/ck/tensor_operation/gpu/grid/gridwise_gemm_wmma_cshuffle_v3.hpp`  (+6/-3)
- `include/ck/tensor_operation/gpu/device/impl/device_gemm_bias_add_reduce_wmma_cshuffle_v3.hpp`  (+4/-1)

## Key added lines (kernel files)

**`include/ck/tensor_operation/gpu/device/impl/device_batched_gemm_wmma_cshuffle_v3.hpp`**
```
using EpilogueType =
typename std::conditional<GridwiseGemm::IsBWaveTransferApplicable &&
GridwiseGemm::UseDirectStore,
typename GridwiseGemm::EpilogueDirectStore,
```

**`include/ck/tensor_operation/gpu/device/impl/device_batched_gemm_wmma_cshuffle_v3_b_scale.hpp`**
```
using EpilogueType =
typename std::conditional<GridwiseGemm::IsBWaveTransferApplicable &&
GridwiseGemm::UseDirectStore,
typename GridwiseGemm::EpilogueDirectStore,
```

**`include/ck/tensor_operation/gpu/device/impl/device_gemm_bias_add_reduce_wmma_cshuffle_v3.hpp`**
```
PermuteB,
false, // IsBPreShuffled
false, // ForceThreadTileTransfer
true>; // IsFusedKernel
```

**`include/ck/tensor_operation/gpu/device/impl/device_gemm_multiple_d_layernorm_wmma_cshuffle_v3.hpp`**
```
PermuteB,
```

**`include/ck/tensor_operation/gpu/device/impl/device_gemm_reduce_wmma_cshuffle_v3.hpp`**
```
PermuteB,
```
