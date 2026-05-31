# Diff summary

- **files changed:** 31
- **lines:** +1401 / -696
- **kernel-ish files:** 31

## Files (by churn)

- `include/ck_tile/ops/gemm/pipeline/gemm_universal_pipeline_ag_bg_cr_policy.hpp`  (+398/-289)
- `include/ck_tile/core/algorithm/static_encoding_pattern.hpp`  (+210/-0)
- `include/ck_tile/core/tensor/transpose_tile.hpp`  (+202/-0)
- `test/ck_tile/gemm/test_gemm_pipeline_util.hpp`  (+97/-71)
- `include/ck_tile/ops/gemm/pipeline/gemm_pipeline_agmem_bgmem_creg_v1_default_policy.hpp`  (+5/-111)
- `include/ck_tile/ops/gemm/kernel/gemm_kernel.hpp`  (+86/-26)
- `include/ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_comp_v3.hpp`  (+85/-24)
- `example/ck_tile/16_batched_gemm/run_batched_gemm_example.inc`  (+59/-26)
- `example/ck_tile/03_gemm/universal_gemm.cpp`  (+61/-7)
- `include/ck_tile/ops/gemm/pipeline/gemm_pipeline_problem.hpp`  (+35/-18)
- `include/ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_base.hpp`  (+26/-16)
- `example/ck_tile/03_gemm/run_gemm_example.inc`  (+0/-36)
- `test/ck_tile/gemm/test_gemm_pipeline_ut_cases.inc`  (+26/-5)
- `test/ck_tile/gemm/test_gemm_pipeline.cpp`  (+15/-13)
- `example/ck_tile/03_gemm/gemm_basic.cpp`  (+23/-3)

## Key added lines (kernel files)

**`example/ck_tile/03_gemm/gemm_basic.cpp`**
```
using CodegenGemmPipeline = ck_tile::GemmPipelineAGmemBGmemCRegV1<CodegenPipelineProblem>;
int run_gemm_example(int argc, char* argv[])
auto [result, arg_parser] = create_args(argc, argv);
if(!result)
```

**`example/ck_tile/03_gemm/universal_gemm.cpp`**
```
constexpr bool TransposeC = false;
using GemmUniversalTraits = ck_tile::
TileGemmUniversalTraits<kPadM, kPadN, kPadK, ALayout, BLayout, CLayout, TransposeC>;
GemmUniversalTraits,
```

**`example/ck_tile/16_batched_gemm/batched_gemm.cpp`**
```
using CodegenGemmPipeline = ck_tile::GemmPipelineAGmemBGmemCRegV1<CodegenPipelineProblem>;
```

**`example/ck_tile/16_batched_gemm/batched_gemm.hpp`**
```
.insert("b_layout", "C", "B tensor data layout - Row by default")
```

**`example/ck_tile/16_batched_gemm/run_batched_gemm_example.inc`**
```
using namespace ck_tile::literals;
auto f_host_tensor_descriptor = [](std::size_t batch_count_,
std::size_t row,
std::size_t col,
```
