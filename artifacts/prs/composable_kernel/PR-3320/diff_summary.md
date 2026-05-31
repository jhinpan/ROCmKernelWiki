# Diff summary

- **files changed:** 9
- **lines:** +13 / -41
- **kernel-ish files:** 9

## Files (by churn)

- `example/ck_tile/38_block_scale_gemm/gemm_utils.hpp`  (+0/-34)
- `example/ck_tile/38_block_scale_gemm/run_gemm_quant_example.inc`  (+4/-3)
- `include/ck_tile/ops/gemm_quant/pipeline/gemm_bquant_pipeline_ag_bg_cr_v3.hpp`  (+2/-2)
- `example/ck_tile/40_streamk_gemm/run_gemm_example.inc`  (+1/-2)
- `include/ck_tile/ops/gemm/pipeline/gemm_universal_pipeline_ag_bg_cr_policy.hpp`  (+2/-0)
- `include/ck_tile/ops/gemm_quant/kernel/grouped_gemm_quant_kernel.hpp`  (+1/-0)
- `include/ck_tile/ops/gemm_quant/pipeline/gemm_aquant_pipeline_ag_bg_cr_mem.hpp`  (+1/-0)
- `include/ck_tile/ops/gemm_quant/pipeline/gemm_wp_bquant_pipeline_ag_bg_cr_base_policy.hpp`  (+1/-0)
- `include/ck_tile/ops/gemm_quant/pipeline/gemm_wp_bquant_pipeline_ag_bg_cr_v2.hpp`  (+1/-0)

## Key added lines (kernel files)

**`example/ck_tile/38_block_scale_gemm/run_gemm_quant_example.inc`**
```
ck_tile::BaseGemmPipelineAgBgCrCompV3<GemmPipelineProblem>,
ck_tile::BaseGemmPipelineAgBgCrMem<GemmPipelineProblem>,
ck_tile::BaseGemmPipelineAgBgCrCompV3<GemmPipelineProblem>>>>;
```

**`example/ck_tile/40_streamk_gemm/run_gemm_example.inc`**
```
stride_C};
```

**`include/ck_tile/ops/gemm_quant/pipeline/gemm_bquant_pipeline_ag_bg_cr_v3.hpp`**
```
Policy::template MakeShuffledARegTileDistribution<Problem>());
Policy::template MakeShuffledBRegTileDistribution<Problem>());
```
