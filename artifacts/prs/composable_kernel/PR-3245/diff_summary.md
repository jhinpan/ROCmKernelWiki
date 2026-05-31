# Diff summary

- **files changed:** 11
- **lines:** +96 / -272
- **kernel-ish files:** 11

## Files (by churn)

- `include/ck_tile/ops/gemm_quant/pipeline/gemm_aquant_pipeline_ag_bg_cr_v3.hpp`  (+1/-67)
- `include/ck_tile/ops/gemm_quant/pipeline/gemm_bquant_pipeline_ag_bg_cr_v3.hpp`  (+3/-63)
- `include/ck_tile/ops/gemm_quant/pipeline/gemm_aquant_pipeline_ag_bg_cr_mem.hpp`  (+2/-61)
- `include/ck_tile/ops/elementwise/unary_element_wise_operation.hpp`  (+35/-23)
- `include/ck_tile/ops/gemm_quant/pipeline/gemm_group_quant_utils.hpp`  (+22/-22)
- `include/ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_comp_v3.hpp`  (+17/-21)
- `include/ck_tile/ops/gemm_quant/kernel/gemm_quant_kernel.hpp`  (+5/-5)
- `test/ck_tile/grouped_gemm_multi_d/test_grouped_gemm_multi_d_util.hpp`  (+6/-1)
- `include/ck_tile/ops/gemm_quant/pipeline/gemm_wp_bquant_pipeline_ag_bg_cr_v2.hpp`  (+3/-3)
- `test/ck_tile/gemm_block_scale/test_gemm_quant_typed.cpp`  (+1/-5)
- `include/ck_tile/ops/gemm_quant/pipeline/gemm_bquant_pipeline_ag_bg_cr_policy.hpp`  (+1/-1)

## Key added lines (kernel files)

**`include/ck_tile/ops/elementwise/unary_element_wise_operation.hpp`**
```
uint32_t sign = a >> 1;
final_sel = (sign & 0x04040404) | 0x03020100;
tmp_pos = __builtin_amdgcn_perm(reg1, reg0, dict_sel);
tmp_neg = __builtin_amdgcn_perm(reg3, reg2, dict_sel);
```

**`include/ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_comp_v3.hpp`**
```
return TailNumber::Odd;
if(tail_number == ck_tile::TailNumber::Odd)
return run_func(
ck_tile::bool_constant<true>{},
```

**`include/ck_tile/ops/gemm_quant/kernel/gemm_quant_kernel.hpp`**
```
make_tuple(integer_divide_ceil(kargs.N, QuantGroupSize::kN), kargs.QK_B),
make_tuple(kargs.stride_BQ, 1),
make_tuple(number<TilePartitioner::NPerBlock / QuantGroupSize::kN>{},
number<TilePartitioner::KPerBlock / QuantGroupSize::kK>{}),
```

**`include/ck_tile/ops/gemm_quant/pipeline/gemm_aquant_pipeline_ag_bg_cr_mem.hpp`**
```
struct AQuantGemmPipelineAgBgCrMem : public BaseGemmPipelineAgBgCrMem<Problem>
```

**`include/ck_tile/ops/gemm_quant/pipeline/gemm_aquant_pipeline_ag_bg_cr_v3.hpp`**
```
struct AQuantGemmPipelineAgBgCrCompV3 : public BaseGemmPipelineAgBgCrCompV3<Problem>
```
