# Diff summary

- **files changed:** 23
- **lines:** +386 / -51
- **kernel-ish files:** 23

## Files (by churn)

- `include/ck/tensor_operation/gpu/grid/gridwise_gemm_wmma_cshuffle_v3_common.hpp`  (+67/-14)
- `include/ck/tensor_operation/gpu/block/thread_group_tensor_slice_transfer_global.hpp`  (+59/-10)
- `include/ck/tensor_operation/gpu/device/impl/device_grouped_gemm_multiple_d_wmma_cshuffle_tile_loop_v3.hpp`  (+23/-0)
- `include/ck/tensor_operation/gpu/device/impl/device_grouped_gemm_wmma_splitk_cshuffle_v3.hpp`  (+22/-1)
- `include/ck/tensor_operation/gpu/device/impl/device_batched_gemm_reduce_wmma_cshuffle_v3.hpp`  (+22/-0)
- `include/ck/tensor_operation/gpu/device/impl/device_gemm_bias_add_reduce_wmma_cshuffle_v3.hpp`  (+22/-0)
- `include/ck/tensor_operation/gpu/device/impl/device_gemm_multiple_d_layernorm_wmma_cshuffle_v3.hpp`  (+22/-0)
- `include/ck/tensor_operation/gpu/device/impl/device_gemm_reduce_wmma_cshuffle_v3.hpp`  (+22/-0)
- `include/ck/tensor_operation/gpu/device/impl/device_batched_contraction_multiple_d_wmma_cshuffle_v3.hpp`  (+20/-0)
- `include/ck/tensor_operation/gpu/device/impl/device_batched_gemm_multiple_d_wmma_cshuffle_v3.hpp`  (+20/-0)
- `include/ck/tensor_operation/gpu/device/impl/device_batched_gemm_wmma_cshuffle_v3_common.hpp`  (+20/-0)
- `include/ck/tensor_operation/gpu/device/impl/device_gemm_wmma_cshuffle_v3_common.hpp`  (+20/-0)
- `include/ck/tensor_operation/gpu/device/impl/device_gemm_wmma_cshuffle_v3r1.hpp`  (+20/-0)
- `example/01_gemm/gemm_wmma_fp16_v3.cpp`  (+5/-5)
- `include/ck/tensor_operation/gpu/device/impl/device_grouped_conv_bwd_data_multiple_d_wmma_cshuffle_v3.hpp`  (+5/-3)

## Key added lines (kernel files)

**`example/01_gemm/gemm_wmma_fp16_v3.cpp`**
```
static constexpr auto GemmSpec = ck::tensor_operation::device::GemmSpecialization::MNKPadding;
PassThrough, PassThrough, PassThrough, GemmSpec,
S<8, 16, 1>, S<0, 2, 1>, S<0, 2, 1>,
1, 8, 8, 1,
```

**`include/ck/tensor_operation/gpu/block/thread_group_tensor_slice_transfer_global.hpp`**
```
oob_thread_scratch_.template SetAsType<bool>(vgpr_data_idx_seq, is_src_valid);
auto index = is_src_valid || !DoTranspose ? src_coord_.GetOffset() : 0;
src_vector_container src_vector = src_vector_container{
grid_buf.template Get<src_vector_container_t, DoTranspose>(index, true)};
```

**`include/ck/tensor_operation/gpu/device/impl/device_batched_contraction_multiple_d_wmma_cshuffle_v3.hpp`**
```
if(ck::is_gfx12_supported() && !GridwiseGemm::CheckValidityAWaveTransfer(arg.M, arg.K))
if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
std::cout << "Wave Transfer not applicable for matrix A" << __FILE__ << ":"
<< __LINE__ << ", in function: " << __func__ << std::endl;
```

**`include/ck/tensor_operation/gpu/device/impl/device_batched_gemm_multiple_d_wmma_cshuffle_v3.hpp`**
```
if(ck::is_gfx12_supported() && !GridwiseGemm::CheckValidityAWaveTransfer(arg.M, arg.K))
if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
std::cout << "Wave Transfer not applicable for matrix A" << __FILE__ << ":"
<< __LINE__ << ", in function: " << __func__ << std::endl;
```

**`include/ck/tensor_operation/gpu/device/impl/device_batched_gemm_reduce_wmma_cshuffle_v3.hpp`**
```
if(ck::is_gfx12_supported() &&
!GridwiseGemm::CheckValidityAWaveTransfer(arg.MRaw_, arg.KRaw_))
if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
std::cout << "Wave Transfer not applicable for matrix A" << __FILE__ << ":"
```
