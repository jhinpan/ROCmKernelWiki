# Diff summary

- **files changed:** 23
- **lines:** +1987 / -272
- **kernel-ish files:** 23

## Files (by churn)

- `include/ck_tile/ops/fmha/pipeline/block_fmha_fwd_splitkv_pipeline_nwarp_sshuffle_qr_ks_vs.hpp`  (+794/-0)
- `include/ck_tile/ops/gemm/warp/warp_gemm_attribute_mfma.hpp`  (+259/-44)
- `include/ck_tile/ops/gemm/warp/warp_gemm_attribute_mfma_impl.hpp`  (+271/-0)
- `include/ck_tile/ops/fmha/pipeline/block_fmha_fwd_splitkv_pipeline_nwarp_sshuffle_qr_ks_vs_default_policy.hpp`  (+226/-0)
- `include/ck_tile/ops/fmha/pipeline/block_fmha_fwd_splitkv_combine_pipeline_default_policy.hpp`  (+126/-47)
- `example/ck_tile/01_fmha/codegen/ops/fmha_fwd_splitkv.py`  (+48/-37)
- `include/ck_tile/ops/fmha/pipeline/block_fmha_fwd_splitkv_combine_pipeline.hpp`  (+65/-18)
- `include/ck_tile/ops/fmha/kernel/fmha_fwd_splitkv_combine_kernel.hpp`  (+37/-19)
- `include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qx_ks_vs_custom_policy.hpp`  (+14/-41)
- `include/ck_tile/ops/gemm/block/block_gemm_areg_bsmem_creg_one_warp_v1.hpp`  (+29/-15)
- `include/ck_tile/ops/gemm/block/block_gemm_areg_bsmem_creg_v2.hpp`  (+29/-15)
- `example/ck_tile/01_fmha/codegen/ops/fmha_fwd.py`  (+24/-18)
- `include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_problem.hpp`  (+29/-7)
- `include/ck_tile/ops/gemm/warp/warp_gemm.hpp`  (+16/-0)
- `include/ck_tile/ops/fmha/kernel/fmha_fwd_splitkv_kernel.hpp`  (+6/-3)

## Key added lines (kernel files)

**`example/ck_tile/01_fmha/codegen/cpp_symbol_map.py`**
```
"qr_nwarp_sshuffle" : "ck_tile::BlockFmhaPipelineEnum::QRKSVS",
```

**`example/ck_tile/01_fmha/codegen/ops/fmha_fwd.py`**
```
ck_tile::sequence<{F_wm0}, {F_wn0}, {F_wk0}>,
ck_tile::sequence<{F_wm1}, {F_wn1}, {F_wk1}>,
F_wm0       : int  # gemm0 warp size along m
F_wn0       : int  # gemm0 warp size along n
```

**`example/ck_tile/01_fmha/codegen/ops/fmha_fwd_splitkv.py`**
```
"qr_nwarp_sshuffle" : "ck_tile::BlockFmhaFwdSplitKVPipelineNWarpSShuffleQRKSVS",
ck_tile::sequence<{F_wm0}, {F_wn0}, {F_wk0}>,
ck_tile::sequence<{F_wm1}, {F_wn1}, {F_wk1}>,
ck_tile::FmhaFwdSplitKVCombineKernel<
```

**`include/ck_tile/core/arch/amd_buffer_addressing.hpp`**
```
(std::is_same<T, fp16_t>::value && (N == 1 || N == 2 || N == 4 || N == 8)) ||
(std::is_same<T, bf16_t>::value && (N == 1 || N == 2 || N == 4 || N == 8)) ||
```

**`include/ck_tile/core/tensor/static_distributed_tensor.hpp`**
```
static_assert(0 < kThreadElementSpaceSize, "Make sure tile distribution is valid");
```
