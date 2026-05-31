# Diff summary

- **files changed:** 34
- **lines:** +338 / -130
- **kernel-ish files:** 15

## Files (by churn)

- `test/ck_tile/gemm_weight_preshuffle/test_gemm_pipeline_util.hpp`  (+71/-15)
- `test/ck_tile/gemm/CMakeLists.txt`  (+40/-30)
- `test/ck_tile/gemm/test_gemm_pipeline_basic_run_test.inc`  (+49/-16)
- `test/ck_tile/gemm_multi_d/test_gemm_multi_d_util.hpp`  (+44/-9)
- `test/ck_tile/batched_gemm/test_batched_gemm_util.hpp`  (+34/-10)
- `test/ck_tile/grouped_gemm/test_grouped_gemm_util.hpp`  (+28/-5)
- `test/ck_tile/gemm/test_gemm_pipeline_smoke_util.hpp`  (+21/-0)
- `example/ck_tile/03_gemm/run_gemm_example.inc`  (+4/-4)
- `example/ck_tile/17_grouped_gemm/grouped_gemm.hpp`  (+4/-4)
- `example/ck_tile/18_flatmm/run_flatmm_example.inc`  (+4/-4)
- `test/ck_tile/gemm/test_gemm_pipeline_universal_run_test.inc`  (+8/-0)
- `test/ck_tile/gemm_weight_preshuffle/test_gemm_pipeline_kernel_types.hpp`  (+4/-2)
- `include/ck_tile/ops/gemm/warp/warp_gemm_attribute_wmma_impl_base_traits.hpp`  (+2/-2)
- `test/ck_tile/moe_sorting/CMakeLists.txt`  (+2/-2)
- `test/ck_tile/permute/test_permute_util.hpp`  (+4/-0)

## Key added lines (kernel files)

**`example/ck_tile/03_gemm/run_gemm_example.inc`**
```
constexpr int kABK1PerLane = 8;
constexpr int kABK0PerLane = GemmConfig::K_Warp_Tile / divisor / kABK1PerLane;
kABK1PerLane});
```

**`example/ck_tile/17_grouped_gemm/grouped_gemm.hpp`**
```
constexpr int kABK1PerLane = 8;
constexpr int kABK0PerLane = GemmConfig::K_Warp_Tile / divisor / kABK1PerLane;
kABK1PerLane});
```

**`example/ck_tile/18_flatmm/run_flatmm_example.inc`**
```
constexpr int kABK1PerLane = 8;
constexpr int kABK0PerLane = FlatmmConfig::K_Warp_Tile / divisor / kABK1PerLane;
kABK1PerLane});
```

**`include/ck_tile/ops/gemm/warp/warp_gemm_attribute_wmma_impl_base_traits.hpp`**
```
static constexpr index_t kABK0PerLane = 1;
static constexpr index_t kABK1PerLane = 8;
```

**`test/ck_tile/batched_gemm/test_batched_gemm_util.hpp`**
```
struct GemmWarpConfig_Mfma
static constexpr ck_tile::index_t M_Tile      = 256;
static constexpr ck_tile::index_t N_Tile      = 256;
static constexpr ck_tile::index_t K_Tile      = 64;
```
