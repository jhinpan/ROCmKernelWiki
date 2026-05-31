# Diff summary

- **files changed:** 9
- **lines:** +63 / -123
- **kernel-ish files:** 8

## Files (by churn)

- `test/ck_tile/gemm_multi_abd/test_gemm_multi_abd_ut_cases_cshuffle.inc`  (+0/-99)
- `include/ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_comp_v3.hpp`  (+10/-7)
- `example/ck_tile/17_grouped_gemm/run_grouped_gemm_example.inc`  (+13/-3)
- `example/ck_tile/17_grouped_gemm/quant_run_grouped_gemm_example.inc`  (+12/-0)
- `include/ck_tile/ops/gemm/kernel/gemm_multi_abd_kernel.hpp`  (+12/-0)
- `test/ck_tile/gemm_multi_abd/test_gemm_multi_abd_default2d.cpp`  (+5/-5)
- `test/ck_tile/gemm_multi_abd/CMakeLists.txt`  (+4/-4)
- `test/ck_tile/gemm_multi_abd/test_gemm_multi_abd_cshuffle.cpp`  (+5/-3)
- `include/ck_tile/ops/gemm/pipeline/gemm_pipeline_problem.hpp`  (+2/-2)

## Key added lines (kernel files)

**`example/ck_tile/17_grouped_gemm/quant_run_grouped_gemm_example.inc`**
```
Ms.clear();
Ns.clear();
Ks.clear();
stride_As.clear();
```

**`example/ck_tile/17_grouped_gemm/run_grouped_gemm_example.inc`**
```
Ms.clear();
Ns.clear();
Ks.clear();
stride_As.clear();
```

**`include/ck_tile/ops/gemm/kernel/gemm_multi_abd_kernel.hpp`**
```
using ADataType = remove_cvref_t<std::tuple_element_t<0, AsDataType>>;
using BDataType = remove_cvref_t<std::tuple_element_t<0, BsDataType>>;
using DDataType = remove_cvref_t<std::tuple_element_t<0, DsDataType>>;
if(ck_tile::get_device_name() == "gfx950" &&
```

**`include/ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_comp_v3.hpp`**
```
block_gemm.LocalPrefetch(
a_lds_gemm_window, b_lds_gemm_window, is_a_load_tr_v, is_b_load_tr_v);
if constexpr(is_a_col_major && !is_a_load_tr_v())
if constexpr(is_b_row_major && !is_b_load_tr_v())
```

**`include/ck_tile/ops/gemm/pipeline/gemm_pipeline_problem.hpp`**
```
if constexpr(std::is_same_v<ALayout, ck_tile::tensor_layout::gemm::ColumnMajor>)
if constexpr(std::is_same_v<BLayout, ck_tile::tensor_layout::gemm::RowMajor>)
```
