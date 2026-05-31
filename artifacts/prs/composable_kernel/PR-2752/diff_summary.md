# Diff summary

- **files changed:** 20
- **lines:** +631 / -436
- **kernel-ish files:** 19

## Files (by churn)

- `test/ck_tile/gemm_multi_d/test_gemm_multi_d_ut_cases.inc`  (+0/-334)
- `test/ck_tile/gemm_multi_d/test_gemm_multi_d_ut_cases_cshuffle.inc`  (+211/-0)
- `test/ck_tile/gemm_multi_d/test_gemm_multi_d_ut_cases_default2d.inc`  (+211/-0)
- `include/ck_tile/ops/epilogue/default_2d_epilogue.hpp`  (+78/-30)
- `test/ck_tile/gemm_multi_d/test_gemm_multi_d_util.hpp`  (+46/-43)
- `test/ck_tile/gemm_multi_d/test_gemm_multi_d_default2d.cpp`  (+43/-0)
- `test/ck_tile/gemm_multi_d/test_gemm_multi_d_cshuffle.cpp`  (+15/-14)
- `include/ck_tile/ops/gemm/kernel/gemm_multi_d_kernel.hpp`  (+6/-0)
- `test/ck_tile/gemm_multi_d/CMakeLists.txt`  (+4/-2)
- `tile_engine/ops/gemm/codegen_utils.py`  (+5/-0)
- `include/ck_tile/ops/fmha/kernel/fmha_bwd_kernel.hpp`  (+2/-2)
- `include/ck_tile/ops/fmha/kernel/fmha_fwd_kernel.hpp`  (+2/-2)
- `include/ck_tile/ops/fmha/kernel/fmha_fwd_pagedkv_kernel.hpp`  (+1/-2)
- `include/ck_tile/ops/fmha/kernel/fmha_fwd_splitkv_combine_kernel.hpp`  (+1/-1)
- `include/ck_tile/ops/fmha/kernel/fmha_fwd_splitkv_kernel.hpp`  (+1/-1)

## Key added lines (kernel files)

**`include/ck_tile/ops/epilogue/default_2d_epilogue.hpp`**
```
static constexpr index_t NumDTensor                    = 0;
typename DsDataType_,
typename DsLayout_,
typename CDElementwise_,
```

**`include/ck_tile/ops/fmha/kernel/fmha_bwd_kernel.hpp`**
```
KGradEpiloguePipeline{}(dk_dram_window, dk_acc_tile, nullptr);
VGradEpiloguePipeline{}(dv_dram_window, dv_acc_tile, nullptr);
```

**`include/ck_tile/ops/fmha/kernel/fmha_fwd_kernel.hpp`**
```
EpiloguePipeline{}(o_dram_window, o_acc_tile, nullptr);
EpiloguePipeline{}(o_dram_window, o_acc_tile, nullptr);
```

**`include/ck_tile/ops/fmha/kernel/fmha_fwd_pagedkv_kernel.hpp`**
```
EpiloguePipeline{}(o_dram_window, o_acc_tile, nullptr);
```

**`include/ck_tile/ops/fmha/kernel/fmha_fwd_splitkv_combine_kernel.hpp`**
```
EpiloguePipeline{}(o_dram_window, o_acc_tile, nullptr);
```
