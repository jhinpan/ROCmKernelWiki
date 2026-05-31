# Diff summary

- **files changed:** 28
- **lines:** +642 / -175
- **kernel-ish files:** 27

## Files (by churn)

- `example/ck_tile/38_block_scale_gemm/run_gemm_quant_example.inc`  (+78/-46)
- `include/ck_tile/core/numeric/pk_fp4.hpp`  (+87/-1)
- `include/ck_tile/host/reference/reference_gemm.hpp`  (+45/-37)
- `test/ck_tile/gemm_block_scale/test_gemm_quant_abquant_a4w4_padding.cpp`  (+65/-0)
- `include/ck_tile/core/utility/mixed_prec_compute_type.hpp`  (+54/-0)
- `include/ck_tile/ops/gemm_quant/pipeline/gemm_wp_abquant_pipeline_ag_bg_cr_v2.hpp`  (+30/-21)
- `test/ck_tile/gemm_block_scale/test_gemm_quant_abquant_a4w4_base.cpp`  (+44/-0)
- `test/ck_tile/gemm_block_scale/test_gemm_quant_abquant_a4w4_preshuffle.cpp`  (+44/-0)
- `include/ck_tile/ops/gemm_quant/pipeline/gemm_abquant_pipeline_ag_bg_cr_v3.hpp`  (+20/-16)
- `example/ck_tile/38_block_scale_gemm/gemm_abquant_quantgrouped.cpp`  (+30/-0)
- `include/ck_tile/ops/gemm_quant/pipeline/gemm_quant_pipeline_problem.hpp`  (+17/-13)
- `include/ck_tile/ops/elementwise/unary_element_wise_operation.hpp`  (+23/-0)
- `include/ck_tile/ops/gemm/pipeline/wp_pipeline_agmem_bgmem_creg_base_policy.hpp`  (+15/-5)
- `include/ck_tile/ops/common/load_interleaved_pk_type.hpp`  (+13/-6)
- `include/ck_tile/ops/gemm_quant/block/block_universal_gemm_as_aquant_bs_bquant_cr.hpp`  (+9/-10)

## Key added lines (kernel files)

**`example/ck_tile/38_block_scale_gemm/gemm_abquant_quantgrouped.cpp`**
```
lut[hash_multiple_strings(
{"fp4", "abquant", "non-preshuffleb", "non-preshufflequant", "1x128x128"})] =
[](const ck_tile::ArgParser& arg_parser) {
using AQuantGroupSize = ck_tile::QuantGroupShape<ck_tile::sequence<1, 1, 128>>;
```

**`example/ck_tile/38_block_scale_gemm/gemm_quant.cpp`**
```
"or bf8i4;  for ABQuant: fp8, bf8, fp4")
```

**`example/ck_tile/38_block_scale_gemm/run_gemm_quant_example.inc`**
```
using ComputeDataType = void;
std::conditional_t<
QuantMode == ck_tile::QuantType::ABQuantGrouped,
ck_tile::BaseGemmPipelineAgBgCrMem<GemmPipelineProblem>,
```

**`include/ck_tile/core/arch/amd_buffer_addressing.hpp`**
```
(std::is_same<T, pk_fp4_t>::value &&
(N == 1 || N == 2 || N == 4 || N == 8 || N == 16 || N == 32)),
```

**`include/ck_tile/core/arch/amd_buffer_addressing_builtins.hpp`**
```
(N == 1 || N == 2 || N == 4 || N == 8 || N == 16 || N == 32))),
```
