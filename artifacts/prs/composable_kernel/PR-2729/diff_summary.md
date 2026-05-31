# Diff summary

- **files changed:** 23
- **lines:** +1838 / -1332
- **kernel-ish files:** 19

## Files (by churn)

- `include/ck_tile/ops/gemm_group_quant/kernel/gemm_bquant_kernel.hpp`  (+0/-679)
- `include/ck_tile/ops/gemm_group_quant/kernel/gemm_quant_kernel.hpp`  (+399/-162)
- `example/ck_tile/38_block_scale_gemm/run_gemm_quant_example.inc`  (+404/-0)
- `example/ck_tile/38_block_scale_gemm/gemm_quant_basic.cpp`  (+376/-0)
- `example/ck_tile/38_block_scale_gemm/gemm_aquant_basic.cpp`  (+0/-226)
- `include/ck_tile/ops/epilogue/cshuffle_epilogue.hpp`  (+151/-52)
- `test/ck_tile/epilogue/test_cshuffle_epilogue_util.hpp`  (+191/-0)
- `include/ck_tile/ops/gemm_group_quant/pipeline/gemm_quant_pipeline_problem.hpp`  (+59/-102)
- `include/ck_tile/host/reference/reference_gemm.hpp`  (+86/-0)
- `test/ck_tile/epilogue/test_cshuffle_epilogue.cpp`  (+84/-0)
- `include/ck_tile/ops/gemm_group_quant/pipeline/tile_gemm_quant_traits.hpp`  (+13/-27)
- `example/ck_tile/38_block_scale_gemm/gemm_utils.hpp`  (+7/-32)
- `test/ck_tile/gemm_block_scale/test_run_gemm_aquant_example.inc`  (+15/-12)
- `example/ck_tile/38_block_scale_gemm/gemm_aquant_preshuffle.cpp`  (+13/-10)
- `example/ck_tile/38_block_scale_gemm/gemm_bquant_basic.cpp`  (+13/-10)

## Key added lines (kernel files)

**`example/ck_tile/38_block_scale_gemm/gemm_aquant_preshuffle.cpp`**
```
float gemm_calc_aquant(const ck_tile::QuantGemmHostArgs& args, const ck_tile::stream_config& s)
using CodegenGemmTraits = ck_tile::TileGemmQuantTraits<kPadM,
GemmConfig::PreshuffleQuant,
ck_tile::QuantType::AQuantGrouped>;
```

**`example/ck_tile/38_block_scale_gemm/gemm_bquant_basic.cpp`**
```
float gemm_calc_bquant(const ck_tile::QuantGemmHostArgs& args, const ck_tile::stream_config& s)
using CodegenGemmTraits = ck_tile::TileGemmQuantTraits<kPadM,
GemmConfig::PreshuffleQuant,
ck_tile::QuantType::BQuantGrouped>;
```

**`example/ck_tile/38_block_scale_gemm/gemm_quant_basic.cpp`**
```
template <typename GemmConfig,
typename TypeConfig,
typename ALayout,
typename BLayout,
```

**`example/ck_tile/38_block_scale_gemm/gemm_utils.hpp`**
```
struct GemmConfigQuant : public GemmConfigBase
static constexpr bool PreshuffleQuant = true;
.insert("v", "1", "0. No validation, 1. Validation on CPU, 2. Validation on GPU")
.insert("prec", "fp8", "data type. fp8/bf8/i4fp8/i4bf8/i4f32fp8/i4f32bf8")
```

**`example/ck_tile/38_block_scale_gemm/run_gemm_aquant_example.inc`**
```
ck_tile::QuantGemmHostArgs args;
args.QK_A      = AQK;
```
