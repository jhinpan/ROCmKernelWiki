# Diff summary

- **files changed:** 7
- **lines:** +220 / -238
- **kernel-ish files:** 7

## Files (by churn)

- `example/ck_tile/17_grouped_gemm/grouped_gemm.cpp`  (+93/-100)
- `example/ck_tile/17_grouped_gemm/grouped_gemm_preshuffle.cpp`  (+79/-83)
- `example/ck_tile/18_flatmm/flatmm_basic.cpp`  (+18/-18)
- `example/ck_tile/03_gemm/gemm_splitk_two_stage_reduce.cpp`  (+13/-14)
- `example/ck_tile/17_grouped_gemm/quant_grouped_gemm.cpp`  (+11/-15)
- `example/ck_tile/03_gemm/gemm_splitk_two_stage_invoker.hpp`  (+3/-4)
- `example/ck_tile/03_gemm/universal_gemm_invoker.hpp`  (+3/-4)

## Key added lines (kernel files)

**`example/ck_tile/03_gemm/gemm_splitk_two_stage_invoker.hpp`**
```
return Run(has_hot_loop_, tail_number_, MemoryOpSet{});
return Run(has_hot_loop_, tail_number_, MemoryOpAtomicAdd{});
return ave_time = BaseGemmPipeline::TailHandler(RunSplitk, has_hot_loop, tail_num);
```

**`example/ck_tile/03_gemm/gemm_splitk_two_stage_reduce.cpp`**
```
return ave_time = ck_tile::launch_kernel_time_mask(
run_flush_cache,
ck_tile::make_kernel<GemmConfig::kBlockPerCu>(
Kernel{}, grids, blocks, 0, kargs));
```

**`example/ck_tile/03_gemm/universal_gemm_invoker.hpp`**
```
return Run(has_hot_loop_, tail_number_, MemoryOpSet{});
return Run(has_hot_loop_, tail_number_, MemoryOpAtomicAdd{});
return ave_time = BaseGemmPipeline::TailHandler(RunSplitk, has_hot_loop, tail_num);
```

**`example/ck_tile/17_grouped_gemm/grouped_gemm.cpp`**
```
const auto Run =
[&](const auto has_hot_loop_, const auto tail_number_, const auto memory_operation_) {
constexpr bool has_hot_loop_v   = has_hot_loop_.value;
constexpr auto tail_number_v    = tail_number_.value;
```

**`example/ck_tile/17_grouped_gemm/grouped_gemm_preshuffle.cpp`**
```
const auto Run =
[&](const auto has_hot_loop_, const auto tail_number_, const auto memory_operation_) {
constexpr bool has_hot_loop_v   = has_hot_loop_.value;
constexpr auto tail_number_v    = tail_number_.value;
```
