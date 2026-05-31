# Diff summary

- **files changed:** 11
- **lines:** +51 / -136
- **kernel-ish files:** 11

## Files (by churn)

- `include/ck_tile/host/kernel_launch.hpp`  (+29/-48)
- `include/ck_tile/host/timer.hpp`  (+4/-73)
- `example/ck_tile/03_gemm/run_gemm_example.inc`  (+8/-5)
- `include/ck_tile/host/stream_config.hpp`  (+4/-1)
- `example/ck_tile/03_gemm/gemm_utils.hpp`  (+2/-1)
- `tile_engine/ops/gemm/benchmark_gemm.cpp`  (+1/-2)
- `tile_engine/ops/gemm/gemm_profiler.hpp`  (+1/-2)
- `example/ck_tile/18_flatmm/flatmm_basic.cpp`  (+1/-1)
- `example/ck_tile/20_grouped_convolution/grouped_convolution_backward_weight.cpp`  (+1/-1)
- `tile_engine/ops/gemm/benchmark_gemm.hpp`  (+0/-1)
- `tile_engine/ops/gemm/gemm_host_api.hpp`  (+0/-1)

## Key added lines (kernel files)

**`example/ck_tile/03_gemm/gemm_utils.hpp`**
```
.insert("flush_cache", "true", "flush cache before running the kernel, defaults to true")
.insert("rotating_count", "1", "rotating count, defaults to 1");
```

**`example/ck_tile/03_gemm/run_gemm_example.inc`**
```
bool flush_cache,
int rotating_count)
nullptr, true, 1, n_warmup, n_repeat, true, flush_cache, rotating_count});
nullptr, true, 1, n_warmup, n_repeat, true, flush_cache, rotating_count});
```

**`example/ck_tile/18_flatmm/flatmm_basic.cpp`**
```
ave_time = ck_tile::launch_kernel_time_mask(
```

**`example/ck_tile/20_grouped_convolution/grouped_convolution_backward_weight.cpp`**
```
float ave_time = ck_tile::launch_kernel_time_mask(
```

**`include/ck_tile/host/kernel_launch.hpp`**
```
template <typename TimerType, typename PreprocessFunc>
CK_TILE_HOST double
preprocess_profiling_impl(TimerType timer, const stream_config& s, PreprocessFunc preprocess)
timer.start(s.stream_id_);
```
