# Diff summary

- **files changed:** 8
- **lines:** +276 / -104
- **kernel-ish files:** 8

## Files (by churn)

- `example/ck_tile/17_grouped_gemm/quant_run_grouped_gemm_example.inc`  (+75/-46)
- `include/ck_tile/ops/gemm_quant/kernel/grouped_gemm_quant_kernel.hpp`  (+38/-20)
- `test/ck_tile/grouped_gemm_quant/test_grouped_gemm_util_quant.hpp`  (+51/-5)
- `include/ck_tile/ops/gemm_quant/pipeline/gemm_bquant_pipeline_ag_bg_cr_v3.hpp`  (+43/-0)
- `example/ck_tile/17_grouped_gemm/quant_grouped_gemm.cpp`  (+25/-14)
- `test/ck_tile/grouped_gemm_quant/test_grouped_gemm_quant_ut_cases.inc`  (+29/-0)
- `example/ck_tile/17_grouped_gemm/quant_grouped_gemm.hpp`  (+11/-16)
- `test/ck_tile/grouped_gemm_quant/test_grouped_gemm_quant.cpp`  (+4/-3)

## Key added lines (kernel files)

**`example/ck_tile/17_grouped_gemm/quant_grouped_gemm.cpp`**
```
ck_tile::QuantType QuantMode = ck_tile::QuantType::BQuantGrouped>
false, // PreshuffleQuant
false, // PreshuffleB
using QuantGemmProblem = typename std::conditional<
```

**`example/ck_tile/17_grouped_gemm/quant_grouped_gemm.hpp`**
```
template <>
struct GemmTypeConfig<ck_tile::bf8_t>
using ADataType   = ck_tile::bf8_t;
using BDataType   = ck_tile::bf8_t;
```

**`example/ck_tile/17_grouped_gemm/quant_run_grouped_gemm_example.inc`**
```
ck_tile::QuantType QuantMode = ck_tile::QuantType::BQuantGrouped,
typename CDEElementWise      = ck_tile::element_wise::PassThrough>
const int group_count                 = arg_parser.get_int("group_count");
const int repeat                      = arg_parser.get_int("repeat");
```

**`include/ck_tile/ops/gemm_quant/kernel/grouped_gemm_quant_kernel.hpp`**
```
if constexpr(kQuantType == QuantType::BQuantGrouped)
const auto& c_block_tile = GemmPipeline{}.template operator()(a_block_window,
b_block_window,
bq_block_window,
```

**`include/ck_tile/ops/gemm_quant/pipeline/gemm_bquant_pipeline_ag_bg_cr_v3.hpp`**
```
template <typename ADramBlockWindowTmp,
typename BDramBlockWindowTmp,
typename BQDramBlockWindowTmp>
CK_TILE_DEVICE auto operator()(const ADramBlockWindowTmp& a_dram_block_window_tmp,
```
