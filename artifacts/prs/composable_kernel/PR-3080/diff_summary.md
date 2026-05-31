# Diff summary

- **files changed:** 13
- **lines:** +2953 / -6
- **kernel-ish files:** 12

## Files (by churn)

- `include/ck_tile/ops/flatmm/pipeline/mx_flatmm_pipeline_agmem_bgmem_creg_v1.hpp`  (+1330/-0)
- `include/ck_tile/ops/flatmm/kernel/mx_flatmm_kernel.hpp`  (+518/-0)
- `example/ck_tile/18_flatmm/mxgemm/mx_flatmm.cpp`  (+506/-0)
- `include/ck_tile/ops/flatmm/pipeline/mx_flatmm_pipeline_agmem_bgmem_creg_v1_policy.hpp`  (+275/-0)
- `example/ck_tile/18_flatmm/mxgemm/run_mx_flatmm.inc`  (+167/-0)
- `include/ck_tile/host/reference/reference_gemm.hpp`  (+87/-0)
- `example/ck_tile/18_flatmm/mxgemm/mxfp4_flatmm.hpp`  (+40/-0)
- `example/ck_tile/18_flatmm/mxgemm/mx_flatmm.hpp`  (+15/-0)
- `include/ck_tile/ops/flatmm/pipeline/flatmm_pipeline_agmem_bgmem_creg_v1_policy.hpp`  (+5/-3)
- `include/ck_tile/ops/flatmm/kernel/flatmm_kernel.hpp`  (+2/-2)
- `example/ck_tile/18_flatmm/CMakeLists.txt`  (+2/-1)
- `include/ck_tile/ops/flatmm.hpp`  (+3/-0)
- `include/ck_tile/ops/gemm/warp/warp_gemm_dispatcher.hpp`  (+3/-0)

## Key added lines (kernel files)

**`example/ck_tile/18_flatmm/mxgemm/mx_flatmm.cpp`**
```
template <typename Layout>
static constexpr inline auto is_row_major(Layout layout_)
return ck_tile::bool_constant<std::is_same_v<ck_tile::remove_cvref_t<decltype(layout_)>,
ck_tile::tensor_layout::gemm::RowMajor>>{};
```

**`example/ck_tile/18_flatmm/mxgemm/mxfp4_flatmm.hpp`**
```
struct MXfp4_FlatmmConfig16
static constexpr ck_tile::index_t M_Tile = 128;
static constexpr ck_tile::index_t N_Tile = 512;
static constexpr ck_tile::index_t K_Tile = 256;
```

**`example/ck_tile/18_flatmm/mxgemm/run_mx_flatmm.inc`**
```
template <typename PrecActType,
typename PrecWeightType,
typename CDataType,
typename FlatmmConfig,
```

**`include/ck_tile/host/reference/reference_gemm.hpp`**
```
template <typename ADataType,
typename BDataType,
typename ScaleDataType,
typename AccDataType,
```

**`include/ck_tile/ops/flatmm/kernel/flatmm_kernel.hpp`**
```
const index_t i_m = amd_wave_read_first_lane(iM * TilePartitioner::MPerBlock);
const index_t i_n = amd_wave_read_first_lane(iN * TilePartitioner::NPerBlock);
```
