# Diff summary

- **files changed:** 12
- **lines:** +371 / -276
- **kernel-ish files:** 9

## Files (by churn)

- `example/ck_tile/18_flatmm/mxgemm/mx_flatmm.cpp`  (+69/-237)
- `example/ck_tile/18_flatmm/mxgemm/mx_flatmm_instance.hpp`  (+172/-0)
- `example/ck_tile/18_flatmm/mxgemm/mx_flatmm_instance.cpp.in`  (+53/-0)
- `include/ck_tile/ops/flatmm/pipeline/flatmm_pipeline_agmem_bgmem_creg_v1.hpp`  (+18/-12)
- `example/ck_tile/18_flatmm/mxgemm/mx_flatmm_instance.cmake`  (+27/-0)
- `example/ck_tile/18_flatmm/mxgemm/mxfp4_flatmm.hpp`  (+20/-0)
- `example/ck_tile/18_flatmm/run_grouped_flatmm_example.inc`  (+0/-17)
- `include/ck_tile/ops/flatmm/pipeline/flatmm_pipeline_agmem_bgmem_creg_v1_policy.hpp`  (+3/-6)
- `example/ck_tile/18_flatmm/CMakeLists.txt`  (+6/-1)
- `example/ck_tile/18_flatmm/moe_flatmm.cpp`  (+1/-1)
- `example/ck_tile/18_flatmm/mxgemm/run_mx_flatmm.inc`  (+1/-1)
- `example/ck_tile/18_flatmm/run_moe_flatmm_example.inc`  (+1/-1)

## Key added lines (kernel files)

**`example/ck_tile/18_flatmm/moe_flatmm.cpp`**
```
auto flatmm_shuffle_b(const ck_tile::HostTensor<T>& t)
```

**`example/ck_tile/18_flatmm/mxgemm/mx_flatmm.cpp`**
```
using FlatmmShape = ck_tile::TileGemmShape<
ck_tile::sequence<FlatmmConfig::M_Tile, FlatmmConfig::N_Tile, FlatmmConfig::K_Tile>,
ck_tile::sequence<FlatmmConfig::M_Warp, FlatmmConfig::N_Warp, FlatmmConfig::K_Warp>,
ck_tile::sequence<FlatmmConfig::M_Warp_Tile,
```

**`example/ck_tile/18_flatmm/mxgemm/mx_flatmm_instance.hpp`**
```
template <typename Layout>
using is_row_major_t = ck_tile::bool_constant<
std::is_same_v<ck_tile::remove_cvref_t<Layout>, ck_tile::tensor_layout::gemm::RowMajor>>;
template <typename FlatmmConfig,
```

**`example/ck_tile/18_flatmm/mxgemm/mxfp4_flatmm.hpp`**
```
template <typename FlatmmConfig,
typename ADataType,
typename BDataType,
typename DsDatatype,
```

**`example/ck_tile/18_flatmm/mxgemm/run_mx_flatmm.inc`**
```
return pass ? 0 : -1;
```
