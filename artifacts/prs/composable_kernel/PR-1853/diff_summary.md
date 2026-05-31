# Diff summary

- **files changed:** 21
- **lines:** +1198 / -254
- **kernel-ish files:** 19

## Files (by churn)

- `include/ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_comp_v4.hpp`  (+559/-0)
- `include/ck_tile/ops/gemm/pipeline/gemm_universal_pipeline_ag_bg_cr_policy.hpp`  (+143/-132)
- `include/ck_tile/core/utility/transpose_vectors.hpp`  (+73/-43)
- `include/ck_tile/ops/gemm/kernel/gemm_kernel.hpp`  (+93/-8)
- `include/ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_comp_v4_default_policy.hpp`  (+92/-0)
- `include/ck_tile/ops/gemm/block/block_gemm_areg_breg_creg_v1.hpp`  (+59/-29)
- `test/ck_tile/gemm/test_gemm_pipeline_util.hpp`  (+71/-16)
- `example/ck_tile/03_gemm/universal_gemm.cpp`  (+42/-6)
- `example/ck_tile/03_gemm/run_gemm_example.inc`  (+14/-3)
- `example/ck_tile/03_gemm/gemm_basic.hpp`  (+11/-5)
- `include/ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_base.hpp`  (+11/-5)
- `test/ck_tile/gemm/test_gemm_pipeline.cpp`  (+10/-5)
- `include/ck_tile/ops/gemm/pipeline/gemm_pipeline_problem.hpp`  (+4/-0)
- `example/ck_tile/03_gemm/CMakeLists.txt`  (+3/-0)
- `include/ck_tile/ops/gemm/pipeline/gemm_pipeline_agmem_bgmem_creg_v1.hpp`  (+3/-0)

## Key added lines (kernel files)

**`example/ck_tile/03_gemm/gemm_basic.hpp`**
```
.insert("split_k", "1", "splitK value")
.insert("init", "0", "0:random, 1:linear, 2:constant(1)");
```

**`example/ck_tile/03_gemm/run_gemm_example.inc`**
```
ck_tile::index_t init_method = arg_parser.get_int("init");
if (init_method == 0) {
ck_tile::FillUniformDistribution<ADataType>{-1.f, 1.f}(a_m_k);
ck_tile::FillUniformDistribution<BDataType>{-1.f, 1.f}(b_k_n);
```

**`example/ck_tile/03_gemm/universal_gemm.cpp`**
```
constexpr bool DoubleSmemBuffer = false;
constexpr bool DoubleSmemBuffer = false;
constexpr ck_tile::index_t M_Tile = 256;
constexpr ck_tile::index_t N_Tile = 256;
```

**`include/ck_tile/core/utility/transpose_vectors.hpp`**
```
static_assert(((NX % 4 == 0 && NY % 4 == 0) || (NX % 2 == 0 && NY % 2 == 0)), "wrong!");
using S2 = array<S, 2>; // typename array<S, 4>::type;
if constexpr(NX % 4 == 0 && NY % 4 == 0)
static_for<0, NY, 4>{}([&](auto iy) {
```

**`include/ck_tile/ops/gemm/block/block_gemm_areg_breg_creg_v1.hpp`**
```
template <typename PipelineProblem_, typename GemmPolicy_>
struct GemmTraits_
using Problem        = remove_cvref_t<PipelineProblem_>;
using Policy         = remove_cvref_t<GemmPolicy_>;
```
