# Diff summary

- **files changed:** 40
- **lines:** +273 / -167
- **kernel-ish files:** 39

## Files (by churn)

- `include/ck_tile/core/arch/amd_buffer_addressing_builtins.hpp`  (+56/-3)
- `include/ck_tile/core/arch/amd_buffer_addressing.hpp`  (+54/-0)
- `include/ck_tile/ops/gemm/kernel/universal_gemm_kernel.hpp`  (+23/-23)
- `include/ck_tile/ops/grouped_convolution/kernel/grouped_convolution_backward_weight_kernel.hpp`  (+17/-19)
- `include/ck_tile/ops/gemm_quant/kernel/gemm_quant_kernel.hpp`  (+13/-13)
- `include/ck_tile/ops/grouped_convolution/kernel/grouped_convolution_forward_kernel.hpp`  (+11/-13)
- `include/ck_tile/ops/gemm/kernel/batched_gemm_kernel.hpp`  (+10/-10)
- `include/ck_tile/ops/grouped_convolution/kernel/grouped_convolution_backward_data_kernel.hpp`  (+9/-9)
- `include/ck_tile/ops/gemm/kernel/grouped_gemm_kernel.hpp`  (+6/-6)
- `include/ck_tile/ops/add_rmsnorm2d_rdquant/pipeline/add_rmsnorm2d_rdquant_fwd_pipeline_three_pass.hpp`  (+4/-4)
- `include/ck_tile/ops/fused_moe/kernel/fused_moegemm_kernel.hpp`  (+4/-4)
- `include/ck_tile/ops/gemm/kernel/gemm_tile_partitioner.hpp`  (+4/-4)
- `include/ck_tile/ops/batched_transpose/kernel/batched_transpose_kernel.hpp`  (+3/-3)
- `include/ck_tile/ops/fmha/kernel/fmha_bwd_kernel.hpp`  (+3/-3)
- `include/ck_tile/ops/fmha/pipeline/block_fmha_fwd_splitkv_pipeline_qr_ks_vs.hpp`  (+3/-3)

## Key added lines (kernel files)

**`include/ck_tile/core/arch/amd_buffer_addressing.hpp`**
```
__device__ inline uint32_t amd_wave_read_first_lane(uint16_t v)
return __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(v));
__device__ inline uint32_t amd_wave_read_first_lane(uint8_t v)
return __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(v));
```

**`include/ck_tile/core/arch/amd_buffer_addressing_builtins.hpp`**
```
T* lds_ptr              = lds_base_ptr + lds_offset;
auto const lds_ptr_sgpr = amd_wave_read_first_lane((reinterpret_cast<uintptr_t>(lds_ptr)));
__device__ inline uint32_t amd_wave_read_first_lane(uint16_t v)
return __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(v));
```

**`include/ck_tile/core/arch/arch.hpp`**
```
return amd_wave_read_first_lane(warp_id);
```

**`include/ck_tile/core/tensor/tile_window.hpp`**
```
amd_wave_read_first_lane(m0_init_value)); // This should be wave independent
```

**`include/ck_tile/ops/add_rmsnorm2d_rdquant/pipeline/add_rmsnorm2d_rdquant_fwd_pipeline_three_pass.hpp`**
```
amd_wave_read_first_lane(integer_divide_ceil(row_size, Block_N));
for(int iN = amd_wave_read_first_lane(0); iN < num_n_tile_iteration; ++iN)
for(int iN = amd_wave_read_first_lane(0); iN < num_n_tile_iteration; ++iN)
for(int iN = amd_wave_read_first_lane(0); iN < num_n_tile_iteration; ++iN)
```
