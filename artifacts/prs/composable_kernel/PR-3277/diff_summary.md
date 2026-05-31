# Diff summary

- **files changed:** 7
- **lines:** +74 / -31
- **kernel-ish files:** 7

## Files (by churn)

- `include/ck_tile/ops/gemm_quant/pipeline/gemm_aquant_pipeline_ag_bg_cr_v3.hpp`  (+49/-9)
- `example/ck_tile/38_block_scale_gemm/run_gemm_quant_example.inc`  (+8/-14)
- `include/ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_base.hpp`  (+6/-4)
- `include/ck_tile/ops/gemm/pipeline/gemm_universal_pipeline_ag_bg_cr_policy.hpp`  (+3/-2)
- `include/ck_tile/ops/gemm_quant/block/block_universal_gemm_as_aquant_bs_cr.hpp`  (+4/-1)
- `include/ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_comp_async_default_policy.hpp`  (+2/-1)
- `test/ck_tile/gemm_block_scale/test_gemm_quant_typed.cpp`  (+2/-0)

## Key added lines (kernel files)

**`example/ck_tile/38_block_scale_gemm/run_gemm_quant_example.inc`**
```
return run_gemm_example_with_layouts<GemmConfig,
TypeConfig,
QuantGroupSize,
QuantMode>(
```

**`include/ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_base.hpp`**
```
template <typename OverrideADataType = ADataType, typename OverrideBDataType = BDataType>
OverrideADataType* __restrict__ p_a_lds = static_cast<OverrideADataType*>(p_smem);
constexpr auto a_lds_block_desc =
Policy::template MakeALdsBlockDescriptor<Problem, OverrideADataType>();
```

**`include/ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_comp_async_default_policy.hpp`**
```
template <typename Problem,
typename OverrideADataType = remove_cvref_t<typename Problem::ADataType>>
```

**`include/ck_tile/ops/gemm/pipeline/gemm_universal_pipeline_ag_bg_cr_policy.hpp`**
```
template <typename Problem,
typename OverrideADataType = remove_cvref_t<typename Problem::ADataType>>
using ADataType             = OverrideADataType;
```

**`include/ck_tile/ops/gemm_quant/block/block_universal_gemm_as_aquant_bs_cr.hpp`**
```
load_int4_tile<BDataType, ComputeDataType, UnaryOpSize_, ALoadTranspose>(
```
