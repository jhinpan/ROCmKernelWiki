# Diff summary

- **files changed:** 8
- **lines:** +429 / -670
- **kernel-ish files:** 8

## Files (by churn)

- `include/ck_tile/ops/gemm_quant/kernel/gemm_quant_kernel.hpp`  (+44/-601)
- `example/ck_tile/38_block_scale_gemm/gemm_bquant_quantgrouped_preshufflequant.cpp`  (+228/-17)
- `include/ck_tile/ops/gemm_quant/pipeline/gemm_group_quant_utils.hpp`  (+128/-38)
- `include/ck_tile/ops/gemm_quant/pipeline/gemm_wp_bquant_pipeline_ag_bg_cr_v2.hpp`  (+14/-6)
- `include/ck_tile/ops/gemm_quant/pipeline/gemm_bquant_pipeline_ag_bg_cr_v3.hpp`  (+10/-5)
- `include/ck_tile/ops/gemm_quant/pipeline/gemm_bquant_pipeline_ag_bg_cr_policy.hpp`  (+3/-2)
- `example/ck_tile/38_block_scale_gemm/run_gemm_quant_example.inc`  (+1/-1)
- `include/ck_tile/ops/gemm_quant/block/block_universal_gemm_as_bs_bquant_cr.hpp`  (+1/-0)

## Key added lines (kernel files)

**`example/ck_tile/38_block_scale_gemm/gemm_bquant_quantgrouped_preshufflequant.cpp`**
```
using TypeConfig     = decltype(GemmQuantTypeConfig<ck_tile::fp8_t,
ck_tile::fp8_t,
ck_tile::half_t,
float>{});
```

**`example/ck_tile/38_block_scale_gemm/run_gemm_quant_example.inc`**
```
ck_tile::host_tensor_descriptor(BQK, BQN, stride_BQ, is_row_major(bq_layout)));
```

**`include/ck_tile/ops/gemm_quant/kernel/gemm_quant_kernel.hpp`**
```
index_t NPerBlockBQ,
MakePreshuffledQuantTensorView(const BQDataType_* bq_ptr, index_t N, index_t QN_B, index_t QK_B)
const auto block_tile_size = NPerBlockBQ * KPerBlockBQ;
const auto bq_pad0_desc = transform_tensor_descriptor(
```

**`include/ck_tile/ops/gemm_quant/pipeline/gemm_bquant_pipeline_ag_bg_cr_policy.hpp`**
```
Problem::BQuantGroupSize::kN,
Problem::BQuantGroupSize::kK,
Problem::BQuantGroupSize::kK,
```

**`include/ck_tile/ops/gemm_quant/pipeline/gemm_bquant_pipeline_ag_bg_cr_v3.hpp`**
```
static constexpr index_t NPerBlockBQ =
integer_divide_ceil(BlockGemmShape::kN, QuantGroupSize::kN);
static constexpr index_t KPerBlockBQ =
integer_divide_ceil(BlockGemmShape::kK, QuantGroupSize::kK);
```
