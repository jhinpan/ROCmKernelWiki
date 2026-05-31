# Diff summary

- **files changed:** 11
- **lines:** +780 / -57
- **kernel-ish files:** 11

## Files (by churn)

- `include/ck_tile/ops/gemm/block/block_universal_gemm_as_bs_cr.hpp`  (+661/-0)
- `include/ck_tile/ops/gemm/warp/warp_gemm_attribute_mfma.hpp`  (+31/-24)
- `include/ck_tile/ops/gemm/pipeline/gemm_pipeline_agmem_bgmem_creg_v1_default_policy.hpp`  (+36/-4)
- `example/ck_tile/03_gemm/gemm_mem_pipeline.cpp`  (+24/-9)
- `example/ck_tile/03_gemm/run_gemm_example.inc`  (+10/-12)
- `include/ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_mem.hpp`  (+6/-6)
- `include/ck_tile/ops/gemm/warp/warp_gemm_impl.hpp`  (+6/-1)
- `example/01_gemm/run_gemm_example_v2.inc`  (+1/-1)
- `include/ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_scheduler.hpp`  (+2/-0)
- `include/ck_tile/ops/gemm/pipeline/gemm_pipeline_problem.hpp`  (+2/-0)
- `include/ck_tile/ops/gemm.hpp`  (+1/-0)

## Key added lines (kernel files)

**`example/01_gemm/run_gemm_example_v2.inc`**
```
invoker.Run(argument, StreamConfig{nullptr, config.time_kernel, 0, 50, 100, true, 4});
```

**`example/ck_tile/03_gemm/gemm_mem_pipeline.cpp`**
```
constexpr ck_tile::index_t N_Tile = 32;
constexpr ck_tile::index_t K_Tile = 64;
constexpr ck_tile::index_t M_Warp = 4;
constexpr ck_tile::index_t N_Warp = 1;
```

**`example/ck_tile/03_gemm/run_gemm_example.inc`**
```
std::cout << "Run Gemm kernel with M =" << M << " N =" << N << " K =" << K
```

**`include/ck_tile/ops/gemm/block/block_universal_gemm_as_bs_cr.hpp`**
```
namespace ck_tile {
template <typename Problem_, typename Policy_ = BlockGemmASmemBSmemCRegV1DefaultPolicy>
struct BlockUniversalGemmAsBsCr
template <typename PipelineProblem_, typename GemmPolicy_>
```

**`include/ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_mem.hpp`**
```
auto block_gemm   = BlockGemm();
auto c_block_tile = block_gemm.MakeCBlockTile();
block_gemm.LocalPrefetch(a_lds_gemm_window, b_lds_gemm_window);
block_gemm.LocalPrefetch(a_lds_gemm_window, b_lds_gemm_window);
```
