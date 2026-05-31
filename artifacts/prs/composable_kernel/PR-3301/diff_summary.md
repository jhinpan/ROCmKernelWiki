# Diff summary

- **files changed:** 22
- **lines:** +465 / -310
- **kernel-ish files:** 19

## Files (by churn)

- `include/ck_tile/utility/json_dump.hpp`  (+238/-237)
- `test/ck_tile/grouped_gemm_preshuffle/test_grouped_gemm_preshuffle_util.hpp`  (+53/-9)
- `test/ck_tile/grouped_gemm_multi_d/test_grouped_gemm_multi_d.cpp`  (+28/-25)
- `test/ck_tile/gemm_multi_abd/test_gemm_multi_abd_util.hpp`  (+33/-3)
- `include/ck_tile/ops/gemm/pipeline/gemm_pipeline_agmem_bgmem_creg_v1.hpp`  (+28/-4)
- `include/ck_tile/ops/gemm/pipeline/gemm_pipeline_agmem_bgmem_creg_v2.hpp`  (+24/-1)
- `test/ck_tile/gemm_weight_preshuffle/test_gemm_pipeline_util.hpp`  (+23/-1)
- `test/ck_tile/gemm_multi_abd/test_gemm_multi_abd_cshuffle.cpp`  (+8/-7)
- `test/ck_tile/grouped_gemm_preshuffle/test_grouped_gemm_preshuffle.cpp`  (+6/-6)
- `test/ck_tile/gemm_multi_abd/test_gemm_multi_abd_default2d.cpp`  (+5/-3)
- `include/ck_tile/ops/gemm/kernel/streamk_gemm_kernel.hpp`  (+4/-3)
- `include/ck_tile/ops/gemm/pipeline/tile_gemm_traits.hpp`  (+3/-2)
- `example/ck_tile/03_gemm/run_gemm_example.inc`  (+2/-2)
- `example/ck_tile/03_gemm/gemm_utils.hpp`  (+1/-1)
- `include/ck_tile/ops/fused_moe/kernel/moe_sorting_kernel.hpp`  (+2/-0)

## Key added lines (kernel files)

**`example/ck_tile/03_gemm/gemm_utils.hpp`**
```
inline auto create_args()
```

**`example/ck_tile/03_gemm/run_gemm_example.inc`**
```
std::tuple<ck_tile::index_t, ck_tile::index_t, ck_tile::index_t> inline parse_gemm_size(
ck_tile::ArgParser& arg_parser)
```

**`include/ck_tile/ops/fused_moe/kernel/moe_sorting_kernel.hpp`**
```
__syncthreads();
```

**`include/ck_tile/ops/gemm/kernel/grouped_gemm_kernel.hpp`**
```
block_sync_lds();
```

**`include/ck_tile/ops/gemm/kernel/streamk_gemm_kernel.hpp`**
```
block_sync_lds();
ck_tile::hip_check_error(hipGetDevice(&dev));
ck_tile::hip_check_error(hipGetDeviceProperties(&dev_prop, dev));
ck_tile::hip_check_error(
```
