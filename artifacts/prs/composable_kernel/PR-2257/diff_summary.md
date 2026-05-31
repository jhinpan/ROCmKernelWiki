# Diff summary

- **files changed:** 17
- **lines:** +415 / -79
- **kernel-ish files:** 16

## Files (by churn)

- `example/ck_tile/03_gemm/universal_gemm.cpp`  (+95/-58)
- `include/ck_tile/host/rotating_buffers.hpp`  (+102/-0)
- `include/ck_tile/host/kernel_launch.hpp`  (+60/-2)
- `example/ck_tile/18_flatmm/flatmm_basic.cpp`  (+42/-5)
- `tile_engine/ops/gemm/gemm_instance_builder.py`  (+44/-1)
- `include/ck_tile/host/flush_icache.hpp`  (+30/-0)
- `tile_engine/ops/gemm/benchmark_gemm.cpp`  (+9/-9)
- `example/ck_tile/03_gemm/gemm_utils.hpp`  (+7/-0)
- `example/ck_tile/18_flatmm/flatmm_basic.hpp`  (+7/-0)
- `tile_engine/ops/gemm/gemm_host_api.hpp`  (+5/-0)
- `include/ck_tile/host/stream_config.hpp`  (+3/-1)
- `tile_engine/ops/gemm/gemm_profiler.hpp`  (+3/-1)
- `example/ck_tile/03_gemm/run_gemm_example.inc`  (+1/-1)
- `example/ck_tile/18_flatmm/run_flatmm_example.inc`  (+1/-1)
- `include/ck_tile/host.hpp`  (+2/-0)

## Key added lines (kernel files)

**`example/ck_tile/03_gemm/gemm_utils.hpp`**
```
template <typename ADataType,
typename BDataType,
typename AccDataType,
typename CDataType,
```

**`example/ck_tile/03_gemm/run_gemm_example.inc`**
```
args, ck_tile::stream_config{nullptr, true, 1, n_warmup, n_repeat, true, true, 50});
```

**`example/ck_tile/03_gemm/universal_gemm.cpp`**
```
const auto Run =
[&](const auto has_hot_loop_, const auto tail_number_, const auto memory_operation_) {
constexpr bool has_hot_loop_v   = has_hot_loop_.value;
constexpr auto tail_number_v    = tail_number_.value;
```

**`example/ck_tile/18_flatmm/flatmm_basic.cpp`**
```
float ave_time{0};
if(s.flush_cache_)
std::cout << "Flushing cache..." << std::endl;
static constexpr ck_tile::index_t APackedSize =
```

**`example/ck_tile/18_flatmm/flatmm_basic.hpp`**
```
template <typename ADataType,
typename BDataType,
typename AccDataType,
typename CDataType,
```
