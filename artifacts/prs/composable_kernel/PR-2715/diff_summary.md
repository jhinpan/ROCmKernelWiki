# Diff summary

- **files changed:** 10
- **lines:** +217 / -151
- **kernel-ish files:** 10

## Files (by churn)

- `include/ck_tile/ops/fmha/pipeline/block_fmha_bwd_dq_dk_dv_pipeline_trload_kr_ktr_vr.hpp`  (+86/-51)
- `include/ck_tile/ops/fmha/pipeline/block_fmha_bwd_dq_dk_dv_pipeline_trload_qr_qtr_dor.hpp`  (+45/-25)
- `include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_async_trload.hpp`  (+21/-20)
- `include/ck_tile/core/arch/amd_buffer_addressing.hpp`  (+22/-17)
- `include/ck_tile/core/arch/amd_buffer_addressing_builtins.hpp`  (+22/-17)
- `include/ck_tile/core/tensor/buffer_view.hpp`  (+10/-10)
- `include/ck_tile/ops/fmha/kernel/fmha_bwd_kernel.hpp`  (+4/-4)
- `include/ck_tile/core/tensor/tensor_view.hpp`  (+3/-3)
- `include/ck_tile/ops/fmha/pipeline/block_fmha_bwd_dq_dk_dv_pipeline_kr_ktr_vr.hpp`  (+2/-2)
- `include/ck_tile/ops/fmha/pipeline/block_fmha_bwd_dq_dk_dv_pipeline_kr_ktr_vr_iglp.hpp`  (+2/-2)

## Key added lines (kernel files)

**`include/ck_tile/core/arch/amd_buffer_addressing.hpp`**
```
llvm_amdgcn_raw_buffer_load_lds(src_wave_buffer_resource,
(as3_uint32_ptr)(smem),
v_offset,
src_wave_addr_offset,
```

**`include/ck_tile/core/arch/amd_buffer_addressing_builtins.hpp`**
```
llvm_amdgcn_raw_buffer_load_lds(src_wave_buffer_resource,
(as3_uint32_ptr)(smem),
v_offset,
src_wave_addr_offset,
```

**`include/ck_tile/core/tensor/buffer_view.hpp`**
```
CK_TILE_HOST_DEVICE constexpr buffer_view(T* __restrict__ p_data, BufferSizeType buffer_size)
CK_TILE_HOST_DEVICE constexpr buffer_view(T* __restrict__ p_data,
CK_TILE_HOST_DEVICE constexpr buffer_view(T* __restrict__ p_data, BufferSizeType buffer_size)
CK_TILE_HOST_DEVICE constexpr buffer_view(T* __restrict__ p_data,
```

**`include/ck_tile/core/tensor/tensor_view.hpp`**
```
CK_TILE_HOST_DEVICE constexpr auto make_tensor_view(DataType* __restrict__ p,
make_naive_tensor_view(DataType* __restrict__ p,
make_naive_tensor_view_packed(DataType* __restrict__ p,
```

**`include/ck_tile/ops/fmha/kernel/fmha_bwd_kernel.hpp`**
```
auto [dk_acc_tile, dv_acc_tile] = FmhaPipeline{}(smem_ptr,
q_dram_window,
FmhaPipeline{}(smem_ptr,
q_dram_window,
```
