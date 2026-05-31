# Diff summary

- **files changed:** 10
- **lines:** +123 / -306
- **kernel-ish files:** 10

## Files (by churn)

- `test/ck_tile/grouped_gemm_preshuffle/test_grouped_gemm_preshuffle_util.hpp`  (+11/-62)
- `include/ck_tile/host/tensor_shuffle_utils.hpp`  (+37/-25)
- `example/ck_tile/03_gemm/gemm_utils.hpp`  (+16/-42)
- `example/ck_tile/17_grouped_gemm/grouped_gemm.hpp`  (+10/-39)
- `example/ck_tile/38_block_scale_gemm/gemm_utils.hpp`  (+9/-39)
- `example/ck_tile/17_grouped_gemm/quant_grouped_gemm.hpp`  (+3/-36)
- `tile_engine/ops/gemm_preshuffle/gemm_preshuffle_common.hpp`  (+0/-39)
- `include/ck_tile/ops/gemm/pipeline/tile_gemm_shape.hpp`  (+22/-0)
- `tile_engine/ops/gemm_preshuffle/gemm_preshuffle_profiler.hpp`  (+15/-6)
- `example/ck_tile/17_grouped_gemm/grouped_gemm_multi_d.hpp`  (+0/-18)

## Key added lines (kernel files)

**`example/ck_tile/03_gemm/gemm_utils.hpp`**
```
static constexpr ck_tile::index_t K_Warp_Tile =
ck_tile::get_k_warp_tile<PrecType, M_Warp_Tile>();
static constexpr ck_tile::index_t K_Warp_Tile =
ck_tile::get_k_warp_tile<PrecType, M_Warp_Tile>();
```

**`example/ck_tile/17_grouped_gemm/grouped_gemm.hpp`**
```
static constexpr ck_tile::index_t K_Warp_Tile =
ck_tile::get_k_warp_tile<PrecType, M_Warp_Tile>();
static constexpr ck_tile::index_t K_Warp_Tile =
ck_tile::get_k_warp_tile<PrecType, M_Warp_Tile>();
```

**`example/ck_tile/17_grouped_gemm/quant_grouped_gemm.hpp`**
```
static constexpr ck_tile::index_t K_Warp_Tile =
ck_tile::get_k_warp_tile<PrecType, M_Warp_Tile>();
ck_tile::get_k_warp_tile<PrecType, M_Warp_Tile, true>();
```

**`example/ck_tile/38_block_scale_gemm/gemm_utils.hpp`**
```
static constexpr ck_tile::index_t K_Warp_Tile =
ck_tile::get_k_warp_tile<PrecType, M_Warp_Tile>();
static constexpr ck_tile::index_t K_Warp_Tile =
ck_tile::get_k_warp_tile<PrecType, M_Warp_Tile>();
```

**`include/ck_tile/host/tensor_shuffle_utils.hpp`**
```
auto shuffle_b(const ck_tile::HostTensor<T>& t, const GemmConfig& gemmConfig)
int kABK0PerLane           = gemmConfig.K_Warp_Tile / divisor / kABK1PerLane;
ck_tile::HostTensor<T> t_view({n_ / gemmConfig.N_Warp_Tile,
gemmConfig.N_Warp_Tile,
```
