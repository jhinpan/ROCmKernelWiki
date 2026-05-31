# Diff summary

- **files changed:** 23
- **lines:** +1260 / -137
- **kernel-ish files:** 20

## Files (by churn)

- `include/ck_tile/ops/gemm_quant/pipeline/gemm_mxfp4_pipeline_ag_bg_cr_v3.hpp`  (+665/-0)
- `include/ck_tile/ops/gemm_quant/pipeline/gemm_mxfp4_pipeline_ag_bg_cr_policy.hpp`  (+140/-0)
- `example/ck_tile/38_block_scale_gemm/run_gemm_quant_example.inc`  (+80/-41)
- `test/ck_tile/gemm_block_scale/test_gemm_quant_fixtures.hpp`  (+71/-38)
- `include/ck_tile/ops/gemm_quant/pipeline/gemm_mxfp4_pipeline_ag_bg_cr_base.hpp`  (+59/-0)
- `include/ck_tile/host/reference/reference_gemm.hpp`  (+57/-0)
- `include/ck_tile/ops/gemm_quant/kernel/gemm_quant_kernel.hpp`  (+35/-14)
- `include/ck_tile/ops/gemm/pipeline/gemm_universal_pipeline_ag_bg_cr_policy.hpp`  (+29/-15)
- `example/ck_tile/38_block_scale_gemm/gemm_bquant_quantgrouped_bf16mxfp4.cpp`  (+41/-0)
- `include/ck_tile/host/check_err.hpp`  (+26/-6)
- `include/ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_base.hpp`  (+14/-7)
- `include/ck_tile/ops/epilogue/cshuffle_epilogue.hpp`  (+10/-4)
- `include/ck_tile/core/arch/amd_buffer_addressing.hpp`  (+4/-3)
- `example/ck_tile/38_block_scale_gemm/gemm_utils.hpp`  (+4/-2)
- `include/ck_tile/ops/gemm/block/block_universal_gemm_as_bs_cr.hpp`  (+4/-2)

## Key added lines (kernel files)

**`example/ck_tile/38_block_scale_gemm/gemm_bquant_quantgrouped_bf16mxfp4.cpp`**
```
template <typename T>
using GemmConfig = GemmConfigQuantPrefill<T>;
run_gemm_example_prec_type<GemmConfig<ck_tile::pk_fp4_raw_t>, \
TypeConfig,                        \
```

**`example/ck_tile/38_block_scale_gemm/gemm_quant.cpp`**
```
"bf8i4 or bf16fp4")
void bquant_quantgrouped_bf16fp4_instance_factory(
std::unordered_map<size_t, std::function<int(const ck_tile::ArgParser&)>>& lut);
bquant_quantgrouped_bf16fp4_instance_factory(lut);
```

**`example/ck_tile/38_block_scale_gemm/gemm_utils.hpp`**
```
using ComputeType = std::conditional_t<
std::is_same_v<BDataType, ck_tile::pk_fp4_raw_t>,
ADataType,
std::conditional_t<sizeof(ADataType) < sizeof(BDataType), ADataType, BDataType>>;
```

**`example/ck_tile/38_block_scale_gemm/run_gemm_quant_example.inc`**
```
std::conditional_t<
GemmConfig::PreshuffleB == true,
ck_tile::WPQuantBPipelineAgBgCrV2<PipelineProblem>,
std::conditional_t<
```

**`include/ck_tile/core/arch/amd_buffer_addressing.hpp`**
```
(N == 1 || N == 2 || N == 4 || N == 8 || N == 16 || N == 32)) ||
(std::is_same<T, pk_fp4_raw_t>::value &&
(N == 1 || N == 2 || N == 4 || N == 8 || N == 16)) ||
(std::is_same<T, pk_fp4_t>::value && (N == 1 || N == 2 || N == 4 || N == 8 || N == 16)),
```
