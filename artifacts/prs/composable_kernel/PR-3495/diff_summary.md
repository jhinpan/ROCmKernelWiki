# Diff summary

- **files changed:** 10
- **lines:** +1161 / -27
- **kernel-ish files:** 8

## Files (by churn)

- `include/ck_tile/ops/gemm_quant/pipeline/gemm_wp_abquant_pipeline_ag_bg_cr_v2.hpp`  (+611/-0)
- `include/ck_tile/ops/gemm_quant/block/block_universal_gemm_ar_aquant_flatbr_bquant_cr.hpp`  (+282/-0)
- `include/ck_tile/ops/gemm_quant/pipeline/gemm_wp_abquant_pipeline_ag_bg_cr_base_policy.hpp`  (+120/-0)
- `example/ck_tile/38_block_scale_gemm/gemm_abquant_quantgrouped.cpp`  (+60/-0)
- `example/ck_tile/38_block_scale_gemm/run_gemm_quant_example.inc`  (+28/-21)
- `test/ck_tile/gemm_block_scale/test_gemm_quant_abquant_preshuffle_2d.cpp`  (+44/-0)
- `test/ck_tile/gemm_block_scale/test_gemm_quant_fixtures.hpp`  (+6/-6)
- `test/ck_tile/gemm_block_scale/CMakeLists.txt`  (+6/-0)
- `include/ck_tile/ops/gemm_quant.hpp`  (+3/-0)
- `CHANGELOG.md`  (+1/-0)

## Key added lines (kernel files)

**`example/ck_tile/38_block_scale_gemm/gemm_abquant_quantgrouped.cpp`**
```
lut[hash_multiple_strings({"fp8",
"abquant",
"preshuffleb",
"non-preshufflequant",
```

**`example/ck_tile/38_block_scale_gemm/run_gemm_quant_example.inc`**
```
std::conditional_t<
QuantMode == ck_tile::QuantType::AQuantGrouped,
ck_tile::BaseGemmPipelineAgBgCrMem<GemmPipelineProblem>,
ck_tile::BaseWeightPreshufflePipelineAGmemBGmemCRegV2<GemmPipelineProblem>>>>;
```

**`include/ck_tile/ops/gemm_quant/block/block_universal_gemm_ar_aquant_flatbr_bquant_cr.hpp`**
```
namespace ck_tile {
template <typename Problem_, typename BlockPolicy_>
struct BlockGemmWeightPreshuffleABQuantARegBRegCReg
template <typename PipelineProblem_, typename GemmPolicy_>
```

**`include/ck_tile/ops/gemm_quant/pipeline/gemm_wp_abquant_pipeline_ag_bg_cr_base_policy.hpp`**
```
namespace ck_tile {
struct GemmWPABQuantPipelineAgBgCrPolicy : public UniversalWeightPreshufflePipelineAgBgCrPolicy
template <typename Problem>
CK_TILE_HOST_DEVICE static constexpr auto GetVectorSizeAQ()
```

**`include/ck_tile/ops/gemm_quant/pipeline/gemm_wp_abquant_pipeline_ag_bg_cr_v2.hpp`**
```
namespace ck_tile {
template <typename Problem, typename PipelinePolicy = GemmWPABQuantPipelineAgBgCrPolicy>
struct WPABQuantBPipelineAgBgCrV2 : public WeightPreshufflePipelineAGmemBGmemCRegV2<Problem>
using Base            = WeightPreshufflePipelineAGmemBGmemCRegV2<Problem>;
```
