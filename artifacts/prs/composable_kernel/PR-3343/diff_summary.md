# Diff summary

- **files changed:** 36
- **lines:** +2421 / -418
- **kernel-ish files:** 32

## Files (by churn)

- `include/ck_tile/ops/gemm_quant/pipeline/gemm_abquant_pipeline_ag_bg_cr_v3.hpp`  (+604/-0)
- `include/ck_tile/ops/gemm_quant/block/block_universal_gemm_as_aquant_bs_bquant_cr.hpp`  (+435/-0)
- `test/ck_tile/gemm_block_scale/test_gemm_quant_fixtures.hpp`  (+308/-0)
- `example/ck_tile/38_block_scale_gemm/run_gemm_quant_example.inc`  (+220/-72)
- `include/ck_tile/ops/gemm_quant/block/block_universal_gemm_as_aquant_bs_cr.hpp`  (+4/-207)
- `include/ck_tile/ops/gemm_quant/block/block_gemm_quant_common.hpp`  (+188/-0)
- `include/ck_tile/host/reference/reference_gemm.hpp`  (+126/-0)
- `experimental/builder/include/ck_tile/builder/factory/conv_dispatcher.hpp`  (+31/-48)
- `example/ck_tile/38_block_scale_gemm/gemm_abquant_quantgrouped.cpp`  (+72/-0)
- `include/ck_tile/ops/gemm_quant/pipeline/gemm_abquant_pipeline_ag_bg_cr_policy.hpp`  (+70/-0)
- `include/ck_tile/ops/gemm_quant/kernel/gemm_quant_kernel.hpp`  (+61/-4)
- `test/ck_tile/gemm_block_scale/test_gemm_quant_abquant.cpp`  (+55/-0)
- `include/ck_tile/ops/gemm_quant/pipeline/gemm_quant_pipeline_problem.hpp`  (+48/-6)
- `include/ck_tile/ops/gemm_quant/block/block_universal_gemm_as_bs_bquant_cr.hpp`  (+9/-36)
- `script/parse_ninja_trace.py`  (+43/-0)

## Key added lines (kernel files)

**`example/ck_tile/38_block_scale_gemm/gemm_abquant_quantgrouped.cpp`**
```
template <typename T>
using GemmConfig = GemmConfigQuantPrefill<T>;
void abquant_quantgrouped_instance_factory(
std::unordered_map<size_t, std::function<int(const ck_tile::ArgParser&)>>& lut)
```

**`example/ck_tile/38_block_scale_gemm/gemm_quant.cpp`**
```
"or bf8i4;  for ABQuant: fp8, bf8")
.insert("quant_mode", "bquant", "Choose aquant, bquant, abquant, tensor or rowcol")
if(quant_mode == "abquant")
std::string preshuffleb =
```

**`example/ck_tile/38_block_scale_gemm/run_gemm_quant_example.inc`**
```
typename AQuantGroupSize,
typename BQuantGroupSize,
std::conditional_t<
QuantMode == ck_tile::QuantType::AQuantGrouped,
```

**`experimental/builder/include/ck_tile/builder/factory/conv_dispatcher.hpp`**
```
concept IsTileAlgorithm = ConvAlgorithmDescriptor<T> && SpecifiesTileThreadBlock<T> &&
SpecifiesTileTransfer<T> && SpecifiesTileConvSpecialization<T> &&
SpecifiesTileBlockGemm<T> && SpecifiesTileOptimizations<T>;
concept IsXdlV3Algorithm =
```

**`experimental/builder/include/ck_tile/builder/testing/type_traits.hpp`**
```
case DataType::FP32_FP32: return 8;
case DataType::FP16_FP16: return 4;
case DataType::BF16_BF16: return 4;
case DataType::BF8: return 1;
```
