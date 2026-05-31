# Diff summary

- **files changed:** 18
- **lines:** +246 / -92
- **kernel-ish files:** 18

## Files (by churn)

- `include/ck_tile/ops/gemm/kernel/gemm_kernel.hpp`  (+120/-44)
- `include/ck_tile/ops/gemm/kernel/batched_gemm_kernel.hpp`  (+25/-7)
- `include/ck_tile/ops/epilogue/cshuffle_epilogue.hpp`  (+27/-4)
- `include/ck_tile/ops/epilogue/default_2d_epilogue.hpp`  (+22/-4)
- `example/ck_tile/03_gemm/universal_gemm.cpp`  (+7/-13)
- `include/ck_tile/ops/gemm/pipeline/gemm_pipeline_agmem_bgmem_creg_v1_default_policy.hpp`  (+8/-6)
- `example/ck_tile/16_batched_gemm/batched_gemm.cpp`  (+9/-4)
- `example/ck_tile/03_gemm/run_gemm_example.inc`  (+4/-4)
- `example/ck_tile/03_gemm/gemm_basic.hpp`  (+3/-3)
- `example/ck_tile/16_batched_gemm/run_batched_gemm_example.inc`  (+4/-0)
- `test/ck_tile/gemm/test_gemm_pipeline_util.hpp`  (+3/-1)
- `example/ck_tile/16_batched_gemm/batched_gemm.hpp`  (+2/-1)
- `test/ck_tile/batched_gemm/test_batched_gemm_util.hpp`  (+2/-1)
- `include/ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_comp_v3.hpp`  (+2/-0)
- `include/ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_mem.hpp`  (+2/-0)

## Key added lines (kernel files)

**`example/ck_tile/03_gemm/gemm_basic.hpp`**
```
arg_parser.insert("m", "3840", "m dimension")
.insert("timer", "gpu", "gpu:gpu timer, cpu:cpu timer")
.insert("split_k", "1", "splitK value");
```

**`example/ck_tile/03_gemm/run_gemm_example.inc`**
```
ck_tile::index_t kbatch = arg_parser.get_int("split_k");
int n_warmup            = arg_parser.get_int("warmup");
int n_repeat            = arg_parser.get_int("repeat");
```

**`example/ck_tile/03_gemm/universal_gemm.cpp`**
```
float gemm_calc(const ck_tile::GemmHostArgs& args, const ck_tile::stream_config& s)
const ck_tile::index_t k_grain     = args.k_batch * K_Tile;
const ck_tile::index_t K_split     = (args.K + k_grain - 1) / k_grain * K_Tile;
const ck_tile::index_t num_loop    = TilePartitioner::GetLoopNum(K_split);
```

**`example/ck_tile/16_batched_gemm/batched_gemm.cpp`**
```
using CodegenGemmPolicy = ck_tile::UniversalGemmPipelineAgBgCrPolicy;
using CodegenGemmPipeline =
ck_tile::GemmPipelineAGmemBGmemCRegV1<CodegenPipelineProblem, CodegenGemmPolicy>;
const dim3 grids      = Kernel::GridSize(args.M, args.N, args.k_batch, args.batch_count);
```

**`example/ck_tile/16_batched_gemm/batched_gemm.hpp`**
```
.insert("timer", "gpu", "gpu:gpu timer, cpu:cpu timer")
.insert("split_k", "1", "splitK value");
```
