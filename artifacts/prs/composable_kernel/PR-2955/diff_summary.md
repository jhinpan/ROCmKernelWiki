# Diff summary

- **files changed:** 8
- **lines:** +924 / -12
- **kernel-ish files:** 7

## Files (by churn)

- `include/ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_comp_v6.hpp`  (+770/-0)
- `include/ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_comp_v6_default_policy.hpp`  (+56/-0)
- `example/ck_tile/03_gemm/gemm_utils.hpp`  (+35/-5)
- `test/ck_tile/gemm/test_gemm_pipeline_kernel_types.hpp`  (+23/-0)
- `test/ck_tile/gemm/test_gemm_pipeline_util.hpp`  (+14/-4)
- `test/ck_tile/gemm/test_gemm_pipeline_compv6.cpp`  (+17/-0)
- `test/ck_tile/gemm/CMakeLists.txt`  (+7/-3)
- `include/ck_tile/ops/gemm.hpp`  (+2/-0)

## Key added lines (kernel files)

**`example/ck_tile/03_gemm/gemm_utils.hpp`**
```
static constexpr bool DoubleSmemBuffer          = false;
static constexpr ck_tile::index_t Pipeline      = CK_TILE_PIPELINE_COMPUTE_V5;
static constexpr ck_tile::index_t NumWaveGroups = 2;
template <typename PrecType>
```

**`include/ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_comp_v6.hpp`**
```
namespace ck_tile {
template <typename Problem>
struct BaseGemmPipelineAgBgCrCompV6
static constexpr index_t PrefetchStages  = 3;
```

**`include/ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_comp_v6_default_policy.hpp`**
```
namespace ck_tile {
struct GemmPipelineAgBgCrCompV6DefaultPolicy
: public UniversalGemmBasePolicy<GemmPipelineAgBgCrCompV6DefaultPolicy>
template <typename Problem>
```

**`test/ck_tile/gemm/test_gemm_pipeline_compv6.cpp`**
```
template <typename T>
class TestCkTileGemmPipelineCompV6
: public TestCkTileGemmPipeline<T, TestCkTileGemmPipelineCompV6<T>>
TYPED_TEST_SUITE(TestCkTileGemmPipelineCompV6, KernelTypesCompV6);
```

**`test/ck_tile/gemm/test_gemm_pipeline_kernel_types.hpp`**
```
using CompV6    = ck_tile::integral_constant<GemmPipelineType, GemmPipelineType::CompV6>;
using KernelTypesCompV6 = ::testing::Types<
std::tuple<    Row,     Row,     Row,       F16,       F16,         F32,       F16,        I256,        I256,         I6
std::tuple<    Col,     Row,     Row,       F16,       F16,         F32,       F16,        I256,        I256,         I6
```
