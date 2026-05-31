# Diff summary

- **files changed:** 41 (diff was byte-capped; summary is partial)
- **lines:** +2454 / -787
- **kernel-ish files:** 40

## Files (by churn)

- `include/ck_tile/ops/fmha/pipeline/block_fmha_fwd_splitkv_pipeline_nwarp_sshuffle_qr_ks_vs.hpp`  (+794/-0)
- `include/ck_tile/ops/gemm/warp/warp_gemm_attribute_mfma.hpp`  (+259/-44)
- `include/ck_tile/ops/gemm/kernel/batched_gemm_kernel.hpp`  (+69/-205)
- `include/ck_tile/ops/gemm/warp/warp_gemm_attribute_mfma_impl.hpp`  (+274/-0)
- `include/ck_tile/ops/gemm/kernel/gemm_kernel.hpp`  (+187/-72)
- `include/ck_tile/ops/fmha/pipeline/block_fmha_fwd_splitkv_pipeline_nwarp_sshuffle_qr_ks_vs_default_policy.hpp`  (+226/-0)
- `include/ck_tile/ops/fmha/pipeline/block_fmha_fwd_splitkv_combine_pipeline_default_policy.hpp`  (+126/-47)
- `include/ck_tile/host/reference/reference_gemm.hpp`  (+11/-151)
- `example/ck_tile/01_fmha/codegen/ops/fmha_fwd_splitkv.py`  (+48/-37)
- `include/ck_tile/ops/fmha/pipeline/block_fmha_fwd_splitkv_combine_pipeline.hpp`  (+65/-18)
- `include/ck_tile/ops/fmha/kernel/fmha_fwd_splitkv_combine_kernel.hpp`  (+37/-19)
- `include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qx_ks_vs_custom_policy.hpp`  (+14/-41)
- `include/ck_tile/host/arg_parser.hpp`  (+44/-2)
- `include/ck_tile/ops/gemm/block/block_gemm_areg_bsmem_creg_one_warp_v1.hpp`  (+29/-15)
- `include/ck_tile/ops/gemm/block/block_gemm_areg_bsmem_creg_v2.hpp`  (+29/-15)

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

**`example/ck_tile/03_gemm/gemm_basic.cpp`**
```
float gemm_calc(const ck_tile::GemmHostArgs& args, const ck_tile::stream_config& s)
auto kargs = Kernel::MakeKernelArgs(args);
const dim3 grids      = Kernel::GridSize(args.M, args.N, args.k_batch);
```

**`example/ck_tile/03_gemm/gemm_basic.hpp`**
```
float gemm_calc(const ck_tile::GemmHostArgs& args, const ck_tile::stream_config& s);
```
