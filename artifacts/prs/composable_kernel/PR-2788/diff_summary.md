# Diff summary

- **files changed:** 39 (diff was byte-capped; summary is partial)
- **lines:** +2838 / -549
- **kernel-ish files:** 33

## Files (by churn)

- `include/ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_mem.hpp`  (+222/-98)
- `example/ck_tile/22_gemm_multi_abd/run_gemm_multi_abd_fp16_example.inc`  (+311/-0)
- `include/ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_comp_v4.hpp`  (+214/-97)
- `include/ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_comp_v3.hpp`  (+181/-72)
- `include/ck_tile/ops/gemm/kernel/gemm_multi_abd_kernel.hpp`  (+193/-0)
- `example/ck_tile/22_gemm_multi_abd/gemm_multi_abd_fp16.hpp`  (+186/-0)
- `example/ck_tile/22_gemm_multi_abd/gemm_multi_abd_fp16.cpp`  (+184/-0)
- `include/ck_tile/ops/gemm/pipeline/gemm_pipeline_agmem_bgmem_creg_v2.hpp`  (+118/-46)
- `test/ck_tile/gemm_multi_abd/test_gemm_multi_abd_ut_cases_cshuffle.inc`  (+160/-0)
- `include/ck_tile/ops/gemm/pipeline/gemm_pipeline_agmem_bgmem_creg_v1.hpp`  (+100/-52)
- `include/ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_comp_v5.hpp`  (+111/-35)
- `include/ck_tile/core/tensor/tile_window.hpp`  (+143/-0)
- `include/ck_tile/ops/gemm/pipeline/gemm_pipeline_problem.hpp`  (+103/-34)
- `include/ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_base.hpp`  (+102/-21)
- `include/ck_tile/host/reference/reference_gemm.hpp`  (+75/-0)

## Key added lines (kernel files)

**`example/ck_tile/22_gemm_multi_abd/gemm_multi_abd_fp16.cpp`**
```
template <typename GemmConfig,
typename AsDataType,
typename BsDataType,
typename DsDataType,
```

**`example/ck_tile/22_gemm_multi_abd/gemm_multi_abd_fp16.hpp`**
```
using A0DataType = ck_tile::half_t;
using A1DataType = ck_tile::half_t;
using B0DataType = ck_tile::half_t;
using B1DataType = ck_tile::half_t;
```

**`example/ck_tile/22_gemm_multi_abd/run_gemm_multi_abd_fp16_example.inc`**
```
template <typename GemmConfig,
typename AsDataType,
typename BsDataType,
typename DsDataType,
```

**`example/ck_tile/22_gemm_multi_abd/utils.hpp`**
```
template <typename Layout>
static constexpr inline auto is_row_major(Layout layout_)
return ck_tile::bool_constant<std::is_same_v<ck_tile::remove_cvref_t<decltype(layout_)>,
ck_tile::tensor_layout::gemm::RowMajor>>{};
```

**`include/ck_tile/core/tensor/load_tile.hpp`**
```
template <typename TileWindow_,
typename ElementWise_,
index_t i_access           = -1,
bool oob_conditional_check = true>
```
