# Diff summary

- **files changed:** 26 (diff was byte-capped; summary is partial)
- **lines:** +4964 / -355
- **kernel-ish files:** 25

## Files (by churn)

- `example/ck_tile/18_flatmm/run_grouped_flatmm_example.inc`  (+605/-0)
- `example/ck_tile/18_flatmm/mixed_prec/a16w4_moe_flatmm.cpp`  (+511/-0)
- `example/ck_tile/18_flatmm/mixed_prec/mixed_prec_flatmm.cpp`  (+482/-0)
- `example/ck_tile/18_flatmm/moe_flatmm.cpp`  (+470/-0)
- `example/ck_tile/18_flatmm/grouped_flatmm.cpp`  (+364/-0)
- `example/ck_tile/18_flatmm/mixed_prec/run_a16w4_moe_flatmm_example.inc`  (+353/-0)
- `example/ck_tile/18_flatmm/flatmm_basic.cpp`  (+283/-51)
- `example/ck_tile/18_flatmm/run_flatmm_example.inc`  (+100/-234)
- `example/ck_tile/18_flatmm/run_moe_flatmm_example.inc`  (+323/-0)
- `include/ck_tile/host/reference/reference_moe_gemm.hpp`  (+316/-0)
- `example/ck_tile/18_flatmm/moe_flatmm.hpp`  (+202/-0)
- `include/ck_tile/core/tensor/tile_scatter_gather.hpp`  (+202/-0)
- `example/ck_tile/18_flatmm/mixed_prec/run_mixed_prec_flatmm.inc`  (+180/-0)
- `include/ck_tile/host/reference/reference_gemm.hpp`  (+177/-0)
- `example/ck_tile/18_flatmm/mixed_prec/a16w4_moe_flatmm.hpp`  (+87/-0)

## Key added lines (kernel files)

**`example/ck_tile/18_flatmm/flatmm_basic.cpp`**
```
template <typename T>
constexpr const char* DataTypeToString()
if constexpr(std::is_same_v<T, ck_tile::half_t>)
return "fp16";
```

**`example/ck_tile/18_flatmm/flatmm_basic.hpp`**
```
static constexpr int kBlockPerCu                = 1;
static constexpr bool TiledMMAPermuteN = false; // disable PermuteN when NWarpTile != 16
static constexpr int kBlockPerCu                = 1;
static constexpr int N_Repeat          = N_Tile / N_Warp_Tile / N_Warp;
```

**`example/ck_tile/18_flatmm/grouped_flatmm.cpp`**
```
template <typename Layout>
static constexpr inline auto is_row_major(Layout layout_)
return ck_tile::bool_constant<std::is_same_v<ck_tile::remove_cvref_t<decltype(layout_)>,
ck_tile::tensor_layout::gemm::RowMajor>>{};
```

**`example/ck_tile/18_flatmm/mixed_prec/a16w4_flatmm.hpp`**
```
struct A16W4_FlatmmConfig16
static constexpr ck_tile::index_t M_Tile = 128;
static constexpr ck_tile::index_t N_Tile = 256;
static constexpr ck_tile::index_t K_Tile = 256;
```

**`example/ck_tile/18_flatmm/mixed_prec/a16w4_moe_flatmm.cpp`**
```
template <typename Layout>
static constexpr inline auto is_row_major(Layout layout_)
return ck_tile::bool_constant<std::is_same_v<ck_tile::remove_cvref_t<decltype(layout_)>,
ck_tile::tensor_layout::gemm::RowMajor>>{};
```
