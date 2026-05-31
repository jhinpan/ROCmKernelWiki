# Diff summary

- **files changed:** 20
- **lines:** +2471 / -26
- **kernel-ish files:** 17

## Files (by churn)

- `include/ck_tile/ops/gemm_group_quant/kernel/gemm_bquant_kernel.hpp`  (+679/-0)
- `include/ck_tile/ops/gemm_group_quant/pipeline/gemm_bquant_pipeline_ag_bg_cr_v3.hpp`  (+475/-0)
- `include/ck_tile/ops/gemm_group_quant/block/block_universal_gemm_as_bs_bquant_cr.hpp`  (+439/-0)
- `example/ck_tile/38_block_scale_gemm/run_gemm_bquant_example.inc`  (+286/-0)
- `example/ck_tile/38_block_scale_gemm/gemm_bquant_basic.cpp`  (+229/-0)
- `include/ck_tile/ops/gemm_group_quant/pipeline/gemm_quant_pipeline_problem.hpp`  (+103/-0)
- `include/ck_tile/ops/gemm_group_quant/pipeline/gemm_bquant_pipeline_ag_bg_cr_policy.hpp`  (+93/-0)
- `include/ck_tile/ops/gemm_group_quant/pipeline/gemm_group_quant_utils.hpp`  (+53/-1)
- `include/ck_tile/ops/gemm_group_quant/pipeline/gemm_bquant_pipeline_ag_bg_cr_base.hpp`  (+53/-0)
- `include/ck_tile/ops/gemm_group_quant/pipeline/tile_gemm_quant_traits.hpp`  (+29/-0)
- `include/ck_tile/ops/gemm_group_quant/block/block_universal_gemm_as_aquant_bs_cr.hpp`  (+8/-8)
- `include/ck_tile/ops/gemm_group_quant.hpp`  (+7/-2)
- `example/ck_tile/38_block_scale_gemm/gemm_aquant_basic.cpp`  (+3/-5)
- `example/ck_tile/38_block_scale_gemm/gemm_utils.hpp`  (+6/-2)
- `include/ck_tile/core/numeric/pk_fp4.hpp`  (+2/-4)

## Key added lines (kernel files)

**`example/ck_tile/38_block_scale_gemm/gemm_aquant_basic.cpp`**
```
s, ck_tile::make_kernel<GemmConfig::kBlockPerCu>(Kernel{}, grids, blocks, 0, kargs));
return run_gemm_example_prec_type<GemmConfig<ck_tile::fp8_t>, TypeConfig, 128>(
return run_gemm_example_prec_type<GemmConfig<ck_tile::bf8_t>, TypeConfig, 128>(
```

**`example/ck_tile/38_block_scale_gemm/gemm_bquant_basic.cpp`**
```
template <typename GemmConfig,
typename ADataType,
typename BDataType,
typename BQDataType,
```

**`example/ck_tile/38_block_scale_gemm/gemm_utils.hpp`**
```
static constexpr int kBlockPerCu           = 1;
static constexpr int kBlockPerCu           = 1;
```

**`example/ck_tile/38_block_scale_gemm/run_gemm_bquant_example.inc`**
```
template <typename Layout>
static constexpr inline auto is_row_major(Layout layout_)
return ck_tile::bool_constant<std::is_same_v<ck_tile::remove_cvref_t<decltype(layout_)>,
ck_tile::tensor_layout::gemm::RowMajor>>{};
```

**`include/ck_tile/core/numeric/pk_fp4.hpp`**
```
return pk_fp4_t::pack(float_to_e2m1(x[0], scale), float_to_e2m1(x[1], scale));
return pk_fp4_t::pack(float_to_e2m1(x[0], scale), float_to_e2m1(x[1], scale));
```
