# Diff summary

- **files changed:** 17
- **lines:** +79 / -414
- **kernel-ish files:** 16

## Files (by churn)

- `example/ck_tile/03_gemm/universal_gemm.cpp`  (+58/-95)
- `include/ck_tile/host/rotating_buffers.hpp`  (+0/-102)
- `include/ck_tile/host/kernel_launch.hpp`  (+2/-59)
- `example/ck_tile/18_flatmm/flatmm_basic.cpp`  (+5/-42)
- `tile_engine/ops/gemm/gemm_instance_builder.py`  (+1/-44)
- `include/ck_tile/host/flush_icache.hpp`  (+0/-30)
- `tile_engine/ops/gemm/benchmark_gemm.cpp`  (+9/-9)
- `example/ck_tile/03_gemm/gemm_utils.hpp`  (+0/-7)
- `example/ck_tile/18_flatmm/flatmm_basic.hpp`  (+0/-7)
- `tile_engine/ops/gemm/gemm_host_api.hpp`  (+0/-5)
- `include/ck_tile/host/stream_config.hpp`  (+1/-3)
- `tile_engine/ops/gemm/gemm_profiler.hpp`  (+1/-3)
- `example/ck_tile/03_gemm/run_gemm_example.inc`  (+1/-1)
- `example/ck_tile/18_flatmm/run_flatmm_example.inc`  (+1/-1)
- `include/ck_tile/host.hpp`  (+0/-2)

## Key added lines (kernel files)

**`example/ck_tile/03_gemm/run_gemm_example.inc`**
```
args, ck_tile::stream_config{nullptr, true, 1, n_warmup, n_repeat});
```

**`example/ck_tile/03_gemm/universal_gemm.cpp`**
```
const auto Run = [&](const auto has_hot_loop_,
const auto tail_number_,
const auto memory_operation_) {
constexpr bool has_hot_loop_v   = has_hot_loop_.value;
```

**`example/ck_tile/18_flatmm/flatmm_basic.cpp`**
```
float ave_time = ck_tile::launch_kernel(
s, ck_tile::make_kernel<blocks.x, kBlockPerCu>(Kernel{}, grids, blocks, 0, kargs));
```

**`example/ck_tile/18_flatmm/run_flatmm_example.inc`**
```
args, ck_tile::stream_config{nullptr, true, 1, n_warmup, n_repeat});
```

**`tile_engine/ops/gemm/benchmark_gemm.cpp`**
```
Setting setting{
arg_parser.get_int("warmup"),
arg_parser.get_int("repeat"),
arg_parser.get_bool("timer"),
```
