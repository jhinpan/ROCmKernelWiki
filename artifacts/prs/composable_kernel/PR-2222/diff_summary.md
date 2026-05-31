# Diff summary

- **files changed:** 10
- **lines:** +234 / -553
- **kernel-ish files:** 10

## Files (by churn)

- `example/ck_tile/16_batched_gemm/batched_gemm.cpp`  (+1/-131)
- `example/ck_tile/17_grouped_gemm/grouped_gemm.cpp`  (+1/-115)
- `example/ck_tile/03_gemm/universal_gemm.cpp`  (+1/-102)
- `test/ck_tile/gemm/test_gemm_pipeline_util.hpp`  (+1/-97)
- `include/ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_comp_v3.hpp`  (+80/-0)
- `include/ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_mem.hpp`  (+76/-2)
- `include/ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_comp_v4.hpp`  (+65/-0)
- `include/ck_tile/ops/gemm/kernel/grouped_gemm_kernel.hpp`  (+7/-54)
- `test/ck_tile/batched_gemm/test_batched_gemm_util.hpp`  (+1/-26)
- `test/ck_tile/grouped_gemm/test_grouped_gemm_util.hpp`  (+1/-26)

## Key added lines (kernel files)

**`example/ck_tile/03_gemm/universal_gemm.cpp`**
```
BaseGemmPipeline::TailHandler(RunSplitk, has_hot_loop, tail_num);
```

**`example/ck_tile/16_batched_gemm/batched_gemm.cpp`**
```
BaseGemmPipeline::TailHandler(RunSplitk, has_hot_loop, tail_num);
```

**`example/ck_tile/17_grouped_gemm/grouped_gemm.cpp`**
```
BaseGemmPipeline::TailHandler(RunSplitk, has_hot_loop, tail_num);
```

**`include/ck_tile/ops/gemm/kernel/grouped_gemm_kernel.hpp`**
```
const auto& c_block_tile = GemmPipeline{}.template operator()(
a_block_window, b_block_window, num_loop, has_hot_loop, tail_num, smem_ptr_0);
auto& c_block_window = gemm_tile_windows.at(Base::I2);
EpiloguePipeline{}.template operator()<decltype(c_block_window), decltype(c_block_tile)>(
```

**`include/ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_comp_v3.hpp`**
```
template <typename RunFunction>
CK_TILE_HOST_DEVICE static auto
TailHandler(const RunFunction& run_func, bool has_hot_loop, TailNumber tail_number)
if(has_hot_loop)
```
