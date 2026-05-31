# Diff summary

- **files changed:** 11
- **lines:** +1798 / -3
- **kernel-ish files:** 8

## Files (by churn)

- `example/ck_tile/17_grouped_gemm/run_grouped_gemm_abquant_example.inc`  (+604/-0)
- `test/ck_tile/grouped_gemm_abquant/test_grouped_gemm_abquant_util.hpp`  (+530/-0)
- `example/ck_tile/17_grouped_gemm/abquant_grouped_gemm.cpp`  (+278/-0)
- `example/ck_tile/17_grouped_gemm/abquant_grouped_gemm.hpp`  (+171/-0)
- `test/ck_tile/grouped_gemm_abquant/test_grouped_gemm_abquant_ut_cases.inc`  (+87/-0)
- `test/ck_tile/grouped_gemm_abquant/test_grouped_gemm_abquant_1x128x128.cpp`  (+47/-0)
- `test/ck_tile/grouped_gemm_abquant/test_grouped_gemm_abquant_1x1x128.cpp`  (+47/-0)
- `include/ck_tile/ops/gemm_quant/kernel/grouped_gemm_quant_kernel.hpp`  (+15/-2)
- `test/ck_tile/grouped_gemm_abquant/CMakeLists.txt`  (+16/-0)
- `example/ck_tile/17_grouped_gemm/CMakeLists.txt`  (+2/-1)
- `test/ck_tile/CMakeLists.txt`  (+1/-0)

## Key added lines (kernel files)

**`example/ck_tile/17_grouped_gemm/abquant_grouped_gemm.cpp`**
```
template <typename GemmConfig,
typename ALayout,
typename AQLayout,
typename BLayout,
```

**`example/ck_tile/17_grouped_gemm/abquant_grouped_gemm.hpp`**
```
template <typename DataType>
struct GemmTypeConfig;
template <>
struct GemmTypeConfig<ck_tile::fp8_t>
```

**`example/ck_tile/17_grouped_gemm/run_grouped_gemm_abquant_example.inc`**
```
template <typename Layout>
static constexpr inline auto is_row_major(Layout layout_)
return ck_tile::bool_constant<std::is_same_v<ck_tile::remove_cvref_t<decltype(layout_)>,
ck_tile::tensor_layout::gemm::RowMajor>>{};
```

**`include/ck_tile/ops/gemm_quant/kernel/grouped_gemm_quant_kernel.hpp`**
```
else if constexpr(kQuantType == QuantType::ABQuantGrouped)
return GemmPipeline{}.template operator()(a_block_window,
b_block_window,
aq_block_window,
```

**`test/ck_tile/grouped_gemm_abquant/test_grouped_gemm_abquant_1x128x128.cpp`**
```
using F16   = ck_tile::half_t;
using F32   = float;
using FP8   = ck_tile::fp8_t;
using BF8   = ck_tile::bf8_t;
```
