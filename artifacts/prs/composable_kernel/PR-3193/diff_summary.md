# Diff summary

- **files changed:** 22
- **lines:** +603 / -289
- **kernel-ish files:** 20

## Files (by churn)

- `test/ck_tile/gemm_block_scale/test_gemm_quant_typed.cpp`  (+0/-146)
- `test/ck_tile/gemm_block_scale/test_gemm_quant_aquant.cpp`  (+95/-0)
- `include/ck_tile/ops/gemm_quant/pipeline/gemm_aquant_pipeline_ag_bg_cr_v3.hpp`  (+52/-26)
- `test/ck_tile/gemm_block_scale/test_gemm_quant_bquant.cpp`  (+77/-0)
- `example/ck_tile/38_block_scale_gemm/run_gemm_quant_example.inc`  (+43/-12)
- `test/ck_tile/gemm_block_scale/test_gemm_quant_bquant_preshuffle.cpp`  (+51/-0)
- `include/ck_tile/ops/gemm_quant/kernel/gemm_quant_kernel.hpp`  (+33/-16)
- `include/ck_tile/ops/gemm_quant/pipeline/gemm_aquant_pipeline_ag_bg_cr_policy.hpp`  (+29/-12)
- `test/ck_tile/gemm_block_scale/test_gemm_quant_rowcol.cpp`  (+38/-0)
- `test/ck_tile/gemm_block_scale/test_gemm_quant_tensor.cpp`  (+38/-0)
- `test/ck_tile/gemm_block_scale/CMakeLists.txt`  (+30/-4)
- `test/ck_tile/gemm_block_scale/test_gemm_quant_ut_cases.inc`  (+0/-33)
- `include/ck_tile/ops/gemm/pipeline/gemm_universal_pipeline_ag_bg_cr_policy.hpp`  (+23/-7)
- `include/ck_tile/ops/gemm_quant/block/block_universal_gemm_as_aquant_bs_cr.hpp`  (+22/-7)
- `include/ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_base.hpp`  (+21/-5)

## Key added lines (kernel files)

**`example/ck_tile/38_block_scale_gemm/run_gemm_quant_example.inc`**
```
typename AQLayout,
typename BQLayout,
AQLayout, // for AQLayout
BQLayout, // for BQLayout
```

**`include/ck_tile/ops/common/load_interleaved_pk_type.hpp`**
```
load_tile(dst, src);
```

**`include/ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_base.hpp`**
```
using WarpTile                  = typename BlockGemmShape::WarpTile;
constexpr index_t kKWarpTile    = WarpTile::at(number<2>{});
constexpr index_t kMaxKWarpTile = (sizeof(ADataType) == 1) ? 64 : 32;
else if constexpr(kKWarpTile > kMaxKWarpTile)
```

**`include/ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_comp_async_default_policy.hpp`**
```
template <typename Problem,
typename OverrideADataType = remove_cvref_t<typename Problem::ADataType>>
```

**`include/ck_tile/ops/gemm/pipeline/gemm_universal_pipeline_ag_bg_cr_policy.hpp`**
```
using ADataType              = remove_cvref_t<typename Problem::ADataType>;
using BDataType              = remove_cvref_t<typename Problem::BDataType>;
using WarpTile               = typename Problem::BlockGemmShape::WarpTile;
constexpr index_t kKWarpTile = WarpTile::at(number<2>{});
```
