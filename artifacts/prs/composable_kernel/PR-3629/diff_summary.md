# Diff summary

- **files changed:** 33
- **lines:** +490 / -367
- **kernel-ish files:** 32

## Files (by churn)

- `include/ck_tile/ops/gemm_quant/kernel/gemm_quant_kernel.hpp`  (+85/-83)
- `example/ck_tile/38_block_scale_gemm/run_gemm_quant_example.inc`  (+21/-24)
- `include/ck_tile/ops/gemm_quant/block/block_universal_gemm_as_bs_bquant_cr.hpp`  (+21/-24)
- `test/ck_tile/gemm_block_scale/test_gemm_quant_abquant_preshuffleQuant.cpp`  (+43/-0)
- `test/ck_tile/gemm_block_scale/CMakeLists.txt`  (+24/-15)
- `include/ck_tile/ops/gemm_quant/pipeline/gemm_aquant_pipeline_ag_bg_cr_policy.hpp`  (+18/-18)
- `include/ck_tile/ops/gemm_quant/block/block_universal_gemm_as_aquant_bs_bquant_cr.hpp`  (+26/-6)
- `include/ck_tile/ops/gemm_quant/pipeline/gemm_bquant_pipeline_ag_bg_cr_v3.hpp`  (+16/-16)
- `example/ck_tile/38_block_scale_gemm/gemm_abquant_quantgrouped.cpp`  (+30/-0)
- `include/ck_tile/ops/gemm_quant/pipeline/gemm_bquant_pipeline_ag_bg_cr_base.hpp`  (+15/-15)
- `include/ck_tile/ops/gemm_quant/pipeline/gemm_mxfp4_pipeline_ag_bg_cr_base.hpp`  (+15/-15)
- `include/ck_tile/ops/gemm_quant/pipeline/gemm_aquant_pipeline_ag_bg_cr_mem.hpp`  (+13/-13)
- `include/ck_tile/ops/gemm_quant/pipeline/gemm_aquant_pipeline_ag_bg_cr_v3.hpp`  (+13/-13)
- `include/ck_tile/ops/gemm_quant/pipeline/gemm_mxfp4_pipeline_ag_bg_cr_v3.hpp`  (+13/-13)
- `include/ck_tile/ops/gemm_quant/pipeline/gemm_wp_bquant_pipeline_ag_bg_cr_v2.hpp`  (+13/-13)

## Key added lines (kernel files)

**`example/ck_tile/17_grouped_gemm/abquant_grouped_gemm.cpp`**
```
false, // APreshuffleQuant
false, // BPreshuffleQuant
false, // APreshuffleQuant
false, // BPreshuffleQuant
```

**`example/ck_tile/17_grouped_gemm/quant_invoke_grouped_gemm_kernel.hpp`**
```
false, // APreshuffleQuant
false, // BPreshuffleQuant
false, // APreshuffleQuant
false, // BPreshuffleQuant
```

**`example/ck_tile/38_block_scale_gemm/gemm_abquant_quantgrouped.cpp`**
```
lut[hash_multiple_strings({"fp8",
"abquant",
"non-preshuffleb",
"preshufflequant",
```

**`example/ck_tile/38_block_scale_gemm/gemm_utils.hpp`**
```
static constexpr bool APreshuffleQuant = false;
static constexpr bool BPreshuffleQuant = false;
static constexpr bool APreshuffleQuant = true;
static constexpr bool BPreshuffleQuant = true;
```

**`example/ck_tile/38_block_scale_gemm/run_gemm_quant_example.inc`**
```
constexpr bool transpose_c =
GemmConfig::TransposeC; // QuantMode == ck_tile::QuantType::ABQuantGrouped;
GemmConfig::APreshuffleQuant,
GemmConfig::BPreshuffleQuant,
```
