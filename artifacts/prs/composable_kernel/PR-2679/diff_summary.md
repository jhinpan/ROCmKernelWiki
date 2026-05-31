# Diff summary

- **files changed:** 10
- **lines:** +279 / -157
- **kernel-ish files:** 10

## Files (by churn)

- `include/ck_tile/ops/gemm_group_quant/block/block_universal_gemm_as_aquant_bs_cr.hpp`  (+183/-142)
- `include/ck_tile/ops/gemm_group_quant/pipeline/gemm_group_quant_utils.hpp`  (+51/-0)
- `include/ck_tile/ops/gemm_group_quant/pipeline/gemm_aquant_pipeline_ag_bg_cr_policy.hpp`  (+26/-12)
- `include/ck_tile/ops/gemm/warp/warp_gemm.hpp`  (+8/-0)
- `include/ck_tile/ops/gemm_group_quant/pipeline/gemm_aquant_pipeline_problem.hpp`  (+4/-1)
- `example/ck_tile/38_block_scale_gemm/gemm_aquant_basic.cpp`  (+2/-1)
- `include/ck_tile/ops/gemm/warp/warp_gemm_dispatcher.hpp`  (+2/-1)
- `example/ck_tile/38_block_scale_gemm/gemm_aquant_preshuffle.cpp`  (+1/-0)
- `include/ck_tile/ops/gemm/warp/warp_gemm_attribute_mfma.hpp`  (+1/-0)
- `test/ck_tile/gemm_block_scale/test_run_gemm_aquant_example.inc`  (+1/-0)

## Key added lines (kernel files)

**`example/ck_tile/38_block_scale_gemm/gemm_aquant_basic.cpp`**
```
constexpr bool transposed_warp_gemm = true;
transposed_warp_gemm,
```

**`example/ck_tile/38_block_scale_gemm/gemm_aquant_preshuffle.cpp`**
```
transposed_warp_gemm,
```

**`include/ck_tile/ops/gemm/warp/warp_gemm.hpp`**
```
using WarpGemmMfma_f32_16x16x32_fp8_fp8_CTransposed =
WarpGemmImpl<WarpGemmAttributeMfmaTransposedCDistribution<
WarpGemmAttributeMfmaImpl_f32_16x16x32_fp8_fp8<WGAttrCtlEnum::Default_>>>;
using WarpGemmMfma_f32_16x16x32_bf8_bf8_CTransposed =
```

**`include/ck_tile/ops/gemm/warp/warp_gemm_attribute_mfma.hpp`**
```
static constexpr index_t kCMLane     = Impl::kCMLane;
```

**`include/ck_tile/ops/gemm/warp/warp_gemm_dispatcher.hpp`**
```
template<> struct WarpGemmDispatcher<ck_tile::fp8_t, ck_tile::fp8_t, float, 16, 16,  32, true> { using Type = WarpGemmMf
template<> struct WarpGemmDispatcher<ck_tile::bf8_t, ck_tile::bf8_t, float, 16, 16,  32, true> { using Type = WarpGemmMf
```
