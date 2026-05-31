# Diff summary

- **files changed:** 13
- **lines:** +803 / -62
- **kernel-ish files:** 11

## Files (by churn)

- `include/ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_comp_async.hpp`  (+531/-0)
- `include/ck_tile/ops/gemm/pipeline/gemm_universal_pipeline_ag_bg_cr_policy.hpp`  (+75/-33)
- `include/ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_comp_async_default_policy.hpp`  (+101/-0)
- `test/ck_tile/gemm/test_gemm_pipeline_ut_cases.inc`  (+33/-18)
- `test/ck_tile/gemm/test_gemm_pipeline_comp_async.cpp`  (+17/-0)
- `include/ck_tile/core/arch/arch.hpp`  (+16/-0)
- `test/ck_tile/gemm/test_gemm_pipeline_util.hpp`  (+13/-2)
- `test/ck_tile/gemm/test_gemm_pipeline_kernel_types.hpp`  (+8/-3)
- `test/ck_tile/gemm/CMakeLists.txt`  (+6/-0)
- `include/ck_tile/ops/gemm_quant/pipeline/gemm_aquant_pipeline_ag_bg_cr_policy.hpp`  (+0/-3)
- `include/ck_tile/ops/gemm_quant/pipeline/gemm_bquant_pipeline_ag_bg_cr_policy.hpp`  (+0/-3)
- `include/ck_tile/ops/gemm.hpp`  (+2/-0)
- `CHANGELOG.md`  (+1/-0)

## Key added lines (kernel files)

**`include/ck_tile/core/arch/arch.hpp`**
```
enum LLVMSchedGroupMask : int32_t
NONE       = 0,
ALU        = 1 << 0,
VALU       = 1 << 1,
```

**`include/ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_comp_async.hpp`**
```
namespace ck_tile {
template <typename Problem>
struct BaseGemmPipelineAgBgCrCompAsync
static constexpr index_t PrefetchStages  = 2;
```

**`include/ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_comp_async_default_policy.hpp`**
```
namespace ck_tile {
struct GemmPipelineAgBgCrCompAsyncDefaultPolicy
: public UniversalGemmBasePolicy<GemmPipelineAgBgCrCompAsyncDefaultPolicy>
static constexpr auto ATileAccessPattern = tile_distribution_pattern::warp_raked;
```

**`include/ck_tile/ops/gemm/pipeline/gemm_universal_pipeline_ag_bg_cr_policy.hpp`**
```
template <typename T, typename = void>
struct has_a_tile_access_pattern : std::false_type
template <typename T>
struct has_a_tile_access_pattern<T, std::void_t<decltype(T::ATileAccessPattern)>> : std::true_type
```

**`test/ck_tile/gemm/test_gemm_pipeline_comp_async.cpp`**
```
template <typename T>
class TestCkTileGemmPipelineCompAsync
: public TestCkTileGemmPipeline<T, class TestCkTileGemmPipelineCompAsync<T>>
TYPED_TEST_SUITE(TestCkTileGemmPipelineCompAsync, KernelTypesCompAsync);
```
