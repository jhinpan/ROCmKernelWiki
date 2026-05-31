# Diff summary

- **files changed:** 13
- **lines:** +182 / -78
- **kernel-ish files:** 13

## Files (by churn)

- `include/ck_tile/host/kernel_launch.hpp`  (+77/-48)
- `include/ck_tile/host/timer.hpp`  (+73/-4)
- `example/ck_tile/03_gemm/run_gemm_example.inc`  (+15/-7)
- `example/ck_tile/03_gemm/gemm_weight_preshuffle.cpp`  (+3/-7)
- `example/ck_tile/03_gemm/universal_gemm.cpp`  (+3/-7)
- `example/ck_tile/03_gemm/gemm_utils.hpp`  (+2/-1)
- `tile_engine/ops/gemm/benchmark_gemm.cpp`  (+2/-1)
- `tile_engine/ops/gemm/gemm_profiler.hpp`  (+2/-1)
- `test/ck_tile/gemm/test_gemm_pipeline_universal_run_test.inc`  (+1/-1)
- `tile_engine/ops/gemm/gemm_instance_builder.py`  (+1/-1)
- `include/ck_tile/host/stream_config.hpp`  (+1/-0)
- `tile_engine/ops/gemm/benchmark_gemm.hpp`  (+1/-0)
- `tile_engine/ops/gemm/gemm_host_api.hpp`  (+1/-0)

## Key added lines (kernel files)

**`example/ck_tile/03_gemm/gemm_utils.hpp`**
```
.insert("persistent", "0", "0:non-persistent, 1:persistent")
.insert("bench_time_ms", "0", "benchmark time in ms, defaults to 0 ms");
```

**`example/ck_tile/03_gemm/gemm_weight_preshuffle.cpp`**
```
auto size_a_buffer = a_m.get_element_space_size_in_bytes();
auto size_b_buffer = b_n.get_element_space_size_in_bytes();
ave_time = ck_tile::launch_kernel_time_mask(
```

**`example/ck_tile/03_gemm/run_gemm_example.inc`**
```
bool persistent,
int bench_time_ms)
ck_tile::stream_config{
nullptr, true, 1, n_warmup, n_repeat, true, true, 50, bench_time_ms});
```

**`example/ck_tile/03_gemm/universal_gemm.cpp`**
```
auto size_a_buffer = a_m.get_element_space_size_in_bytes();
auto size_b_buffer = b_n.get_element_space_size_in_bytes();
ave_time = ck_tile::launch_kernel_time_mask(
```

**`include/ck_tile/host/kernel_launch.hpp`**
```
template <class it>
typename std::iterator_traits<it>::value_type median(it begin, it end)
if(begin == end)
return std::numeric_limits<double>::quiet_NaN();
```
