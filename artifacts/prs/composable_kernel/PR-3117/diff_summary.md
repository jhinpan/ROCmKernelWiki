# Diff summary

- **files changed:** 13
- **lines:** +220 / -238
- **kernel-ish files:** 13

## Files (by churn)

- `example/ck_tile/03_gemm/gemm_utils.hpp`  (+41/-48)
- `example/ck_tile/17_grouped_gemm/grouped_gemm.hpp`  (+31/-36)
- `example/ck_tile/20_grouped_convolution/conv_configs.hpp`  (+26/-31)
- `test/ck_tile/gemm/test_gemm_pipeline_smoke_util.hpp`  (+24/-29)
- `example/ck_tile/17_grouped_gemm/grouped_gemm_multi_d.hpp`  (+21/-25)
- `example/ck_tile/22_gemm_multi_abd/gemm_multi_abd_fp16.hpp`  (+17/-24)
- `example/ck_tile/19_gemm_multi_d/gemm_multi_d_fp16.hpp`  (+17/-20)
- `example/ck_tile/16_batched_gemm/batched_gemm.hpp`  (+16/-20)
- `include/ck_tile/ops/gemm/pipeline/gemm_pipelines.hpp`  (+21/-0)
- `example/ck_tile/03_gemm/universal_gemm.cpp`  (+3/-3)
- `example/ck_tile/03_gemm/gemm_basic.cpp`  (+1/-1)
- `example/ck_tile/03_gemm/gemm_splitk_two_stage_reduce.cpp`  (+1/-1)
- `include/ck_tile/ops/gemm.hpp`  (+1/-0)

## Key added lines (kernel files)

**`example/ck_tile/03_gemm/gemm_basic.cpp`**
```
if constexpr(GemmConfig::Pipeline == ck_tile::GemmPipeline::COMPUTE_V3)
```

**`example/ck_tile/03_gemm/gemm_splitk_two_stage_reduce.cpp`**
```
if constexpr(GemmConfig<ck_tile::half_t>::Pipeline == ck_tile::GemmPipeline::COMPUTE_V3)
```

**`example/ck_tile/03_gemm/gemm_utils.hpp`**
```
static constexpr ck_tile::GemmPipeline Pipeline = ck_tile::GemmPipeline::COMPUTE_V3;
static constexpr bool DoubleSmemBuffer          = false;
static constexpr ck_tile::GemmPipeline Pipeline = ck_tile::GemmPipeline::MEMORY;
static constexpr auto Scheduler                 = ck_tile::GemmPipelineScheduler::Interwave;
```

**`example/ck_tile/03_gemm/universal_gemm.cpp`**
```
if constexpr(GemmConfig<ck_tile::half_t>::Pipeline == ck_tile::GemmPipeline::COMPUTE_V3)
if constexpr(GemmConfig<ck_tile::fp8_t>::Pipeline == ck_tile::GemmPipeline::COMPUTE_V3)
if constexpr(GemmConfig<ck_tile::bf8_t>::Pipeline == ck_tile::GemmPipeline::COMPUTE_V3)
```

**`example/ck_tile/16_batched_gemm/batched_gemm.hpp`**
```
static constexpr bool DoubleSmemBuffer          = false;
static constexpr ck_tile::GemmPipeline Pipeline = ck_tile::GemmPipeline::MEMORY;
static constexpr auto Scheduler                 = ck_tile::GemmPipelineScheduler::Interwave;
static constexpr bool DoubleSmemBuffer          = false;
```
