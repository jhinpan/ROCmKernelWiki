# Diff summary

- **files changed:** 36
- **lines:** +411 / -491
- **kernel-ish files:** 34

## Files (by churn)

- `test/ck_tile/gemm/test_gemm_pipeline_universal_run_test.inc`  (+43/-144)
- `test/ck_tile/gemm/test_gemm_pipeline_basic_run_test.inc`  (+44/-134)
- `test/ck_tile/gemm/test_gemm_pipeline_type_param_product.hpp`  (+63/-0)
- `test/ck_tile/gemm/test_gemm_pipeline_kernel_types.hpp`  (+17/-37)
- `test/ck_tile/gemm/test_gemm_pipeline_smoke_run_test.inc`  (+19/-25)
- `test/ck_tile/gemm/test_gemm_pipeline_smoke_util.hpp`  (+18/-25)
- `test/ck_tile/gemm/test_gemm_pipeline_basic_cases.hpp`  (+25/-0)
- `test/ck_tile/gemm/test_gemm_pipeline_universal_cases.hpp`  (+25/-0)
- `test/ck_tile/gemm/CMakeLists.txt`  (+12/-12)
- `test/ck_tile/gemm/test_gemm_pipeline_universal_fp8.cpp`  (+10/-10)
- `test/ck_tile/gemm/test_gemm_pipeline_basic_fp16.cpp`  (+9/-10)
- `test/ck_tile/gemm/test_gemm_pipeline_basic_fp8.cpp`  (+10/-9)
- `test/ck_tile/gemm/test_gemm_pipeline_universal_bf8.cpp`  (+9/-10)
- `test/ck_tile/gemm/test_gemm_pipeline_basic_bf8.cpp`  (+9/-9)
- `test/ck_tile/gemm/test_gemm_pipeline_universal_bf16.cpp`  (+9/-9)

## Key added lines (kernel files)

**`include/ck_tile/ops/elementwise/unary_element_wise_operation.hpp`**
```
CK_TILE_HOST_DEVICE auto operator()(E& e, const C& c, const Ds&...) const -> void
```

**`include/ck_tile/ops/gemm/block/block_universal_gemm_as_bs_cr.hpp`**
```
const ASmemBlockWindow&,
const BSmemBlockWindow&,
```

**`include/ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_comp_v4.hpp`**
```
if constexpr(HasHotLoop)
static_for<0, 8, 1>{}([&](auto) {
```

**`include/ck_tile/ops/gemm/warp/warp_gemm.hpp`**
```
using WarpGemmMfma_f32_16x16x32_fp8_bf8 = WarpGemmImpl<
WarpGemmAttributeMfma<WarpGemmAttributeMfmaImpl_f32_16x16x32_fp8_bf8<WGAttrCtlEnum::Default_>>>;
using WarpGemmMfma_f32_32x32x32_fp8_bf8 = WarpGemmImpl<WarpGemmAttributeMfmaIterateK<
WarpGemmAttributeMfmaImpl_f32_32x32x16_fp8_bf8<WGAttrCtlEnum::Default_>,
```

**`include/ck_tile/ops/gemm/warp/warp_gemm_attribute_mfma_impl.hpp`**
```
template <WGAttrCtlEnum Ctrl_ = WGAttrCtlEnum::Default_>
using WarpGemmAttributeMfmaImpl_f32_16x16x32_fp8_bf8 =
WarpGemmAttributeMfmaImpl_f32_16x16x32_f8_base<fp8_t, bf8_t, Ctrl_>;
```
