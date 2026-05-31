# Diff summary

- **files changed:** 12
- **lines:** +280 / -123
- **kernel-ish files:** 12

## Files (by churn)

- `example/ck_tile/03_gemm/universal_gemm.cpp`  (+34/-84)
- `example/ck_tile/03_gemm/run_gemm_example.inc`  (+84/-7)
- `example/ck_tile/03_gemm/gemm_utils.hpp`  (+71/-6)
- `include/ck_tile/ops/gemm/kernel/gemm_kernel.hpp`  (+55/-12)
- `include/ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_comp_v3.hpp`  (+11/-8)
- `include/ck_tile/ops/gemm/pipeline/tile_gemm_shape.hpp`  (+8/-1)
- `example/ck_tile/17_grouped_gemm/grouped_gemm.hpp`  (+4/-4)
- `include/ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_comp_v4.hpp`  (+3/-0)
- `include/ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_mem.hpp`  (+3/-0)
- `include/ck_tile/ops/gemm/pipeline/gemm_pipeline_agmem_bgmem_creg_v1.hpp`  (+3/-0)
- `include/ck_tile/ops/gemm/pipeline/gemm_pipeline_agmem_bgmem_creg_v2.hpp`  (+3/-0)
- `example/ck_tile/03_gemm/gemm_basic.cpp`  (+1/-1)

## Key added lines (kernel files)

**`example/ck_tile/03_gemm/gemm_utils.hpp`**
```
struct GemmConfig
static constexpr ck_tile::index_t M_Tile = 128;
static constexpr ck_tile::index_t N_Tile = 32;
static constexpr ck_tile::index_t K_Tile = 64;
```

**`example/ck_tile/03_gemm/run_gemm_example.inc`**
```
template <typename Tensor,
typename ADataType,
typename BDataType,
typename AccDataType,
```

**`example/ck_tile/03_gemm/universal_gemm.cpp`**
```
using GemmShape = ck_tile::TileGemmShape<
ck_tile::sequence<GemmConfig::M_Tile, GemmConfig::N_Tile, GemmConfig::K_Tile>,
ck_tile::sequence<GemmConfig::M_Warp, GemmConfig::N_Warp, GemmConfig::K_Warp>,
ck_tile::
```

**`example/ck_tile/17_grouped_gemm/grouped_gemm.hpp`**
```
struct GemmTypeConfig;
struct GemmTypeConfig<ck_tile::half_t>
using Types = GemmTypeConfig<ck_tile::half_t>;
```

**`include/ck_tile/ops/gemm/kernel/gemm_kernel.hpp`**
```
static_assert(!TilePartitioner::BlockGemmShape::PermuteA, "Not implemented!");
if constexpr(TilePartitioner::BlockGemmShape::PermuteB)
constexpr index_t K1          = GemmPipeline::GetSmemPackB();
const index_t K0              = splitk_batch_offset.splitted_k / K1;
```
