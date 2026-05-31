# Diff summary

- **files changed:** 10
- **lines:** +361 / -191
- **kernel-ish files:** 10

## Files (by churn)

- `include/ck_tile/ops/gemm_quant/pipeline/gemm_group_quant_utils.hpp`  (+103/-52)
- `test/ck_tile/gemm_block_scale/test_gemm_quant_bquant_preshuffle.cpp`  (+45/-45)
- `include/ck_tile/ops/gemm_quant/pipeline/gemm_bquant_pipeline_ag_bg_cr_v3.hpp`  (+61/-24)
- `test/ck_tile/gemm_block_scale/test_gemm_quant_bquant.cpp`  (+46/-30)
- `include/ck_tile/ops/gemm_quant/kernel/gemm_quant_kernel.hpp`  (+49/-17)
- `include/ck_tile/ops/gemm_quant/block/block_universal_gemm_as_bs_bquant_cr.hpp`  (+25/-7)
- `include/ck_tile/ops/gemm_quant/pipeline/gemm_bquant_pipeline_ag_bg_cr_policy.hpp`  (+16/-6)
- `include/ck_tile/ops/gemm_quant/pipeline/gemm_bquant_pipeline_ag_bg_cr_base.hpp`  (+9/-5)
- `test/ck_tile/gemm_block_scale/test_gemm_quant_fixtures.hpp`  (+5/-3)
- `test/ck_tile/gemm_block_scale/test_gemm_quant_base.hpp`  (+2/-2)

## Key added lines (kernel files)

**`include/ck_tile/ops/gemm_quant/block/block_universal_gemm_as_bs_bquant_cr.hpp`**
```
using BQLayout        = remove_cvref_t<typename Problem::BQLayout>;
using OverrideBDataType =
std::conditional_t<std::is_same_v<BDataType, pk_int4_t>, ADataType, BDataType>;
template <typename ASmemBlockWindow,
```

**`include/ck_tile/ops/gemm_quant/kernel/gemm_quant_kernel.hpp`**
```
static_assert(std::is_same_v<BQLayout, tensor_layout::gemm::ColumnMajor>,
"PreshuffleQuant with BQuantGrouped currently only supports "
"ColumnMajor BQ layout");
if constexpr(std::is_same_v<BQLayout, tensor_layout::gemm::RowMajor>)
```

**`include/ck_tile/ops/gemm_quant/pipeline/gemm_bquant_pipeline_ag_bg_cr_base.hpp`**
```
using YPerTile =
std::conditional_t<std::is_same_v<BQLayout, tensor_layout::gemm::ColumnMajor>,
number<NPerBlockBQ>,
number<KPerBlockBQ>>;
```

**`include/ck_tile/ops/gemm_quant/pipeline/gemm_bquant_pipeline_ag_bg_cr_policy.hpp`**
```
if constexpr(std::is_same_v<BQLayout, ck_tile::tensor_layout::gemm::RowMajor>)
return GetABQGlobalVectorLoadSize<Problem, BQDataType, KPerBlockBQ, NPerBlockBQ>();
static_assert(std::is_same_v<BQLayout, ck_tile::tensor_layout::gemm::ColumnMajor>);
return GetABQGlobalVectorLoadSize<Problem, BQDataType, NPerBlockBQ, KPerBlockBQ>();
```

**`include/ck_tile/ops/gemm_quant/pipeline/gemm_bquant_pipeline_ag_bg_cr_v3.hpp`**
```
using OverrideBDataType =
std::conditional_t<std::is_same_v<BDataType, pk_int4_t>, ADataType, BDataType>;
static constexpr auto is_a_load_tr_v = bool_constant<PipelineImplBase::is_a_load_tr>{};
static constexpr auto is_b_load_tr_v = bool_constant<PipelineImplBase::is_b_load_tr>{};
```
