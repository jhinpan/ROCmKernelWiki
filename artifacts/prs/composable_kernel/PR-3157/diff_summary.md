# Diff summary

- **files changed:** 15
- **lines:** +2530 / -19
- **kernel-ish files:** 11

## Files (by churn)

- `tile_engine/ops/gemm_streamk/gemm_streamk_instance_builder.py`  (+905/-0)
- `tile_engine/ops/gemm_streamk/gemm_streamk_validation_utils.py`  (+350/-0)
- `tile_engine/ops/gemm_streamk/gemm_streamk_profiler.hpp`  (+296/-0)
- `tile_engine/ops/gemm_streamk/CMakeLists.txt`  (+295/-0)
- `tile_engine/ops/gemm_streamk/gemm_streamk_benchmark.hpp`  (+201/-0)
- `tile_engine/ops/gemm_streamk/gemm_streamk_benchmark_single.cpp`  (+169/-0)
- `tile_engine/ops/gemm_streamk/gemm_streamk_common.hpp`  (+145/-0)
- `tile_engine/ops/gemm_streamk/configs/default_config.json`  (+105/-0)
- `tile_engine/include/utility/validation.hpp`  (+50/-0)
- `include/ck_tile/ops/gemm/kernel/streamk_gemm_kernel.hpp`  (+2/-11)
- `Jenkinsfile`  (+6/-2)
- `example/ck_tile/40_streamk_gemm/streamk_gemm_basic.cpp`  (+2/-2)
- `test/ck_tile/gemm_streamk/test_gemm_streamk_util.hpp`  (+1/-2)
- `tile_engine/ops/CMakeLists.txt`  (+2/-1)
- `example/ck_tile/40_streamk_gemm/run_gemm_example.inc`  (+1/-1)

## Key added lines (kernel files)

**`example/ck_tile/40_streamk_gemm/run_gemm_example.inc`**
```
if(reduction_strategy == ck_tile::StreamKReductionStrategy::Atomic)
```

**`example/ck_tile/40_streamk_gemm/streamk_gemm_basic.cpp`**
```
if constexpr(ReductionStrategy == ck_tile::StreamKReductionStrategy::Atomic)
else if constexpr(ReductionStrategy == ck_tile::StreamKReductionStrategy::Reduction)
```

**`include/ck_tile/ops/gemm/kernel/streamk_gemm_kernel.hpp`**
```
index_t stride_C_)
stride_C_)
```

**`test/ck_tile/gemm_streamk/test_gemm_streamk_util.hpp`**
```
stride_C};
```

**`tile_engine/include/utility/validation.hpp`**
```
template <typename ADataType, typename BDataType, typename AccDataType, typename CDataType>
auto calculate_rtol_atol(const ck_tile::index_t K,
const ck_tile::index_t kbatch,
const float max_accumulated_value)
```
