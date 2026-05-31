# Diff summary

- **files changed:** 17
- **lines:** +698 / -595
- **kernel-ish files:** 14

## Files (by churn)

- `include/ck_tile/ops/flatmm/pipeline/mx_flatmm_pipeline_agmem_bgmem_creg_v1.hpp`  (+191/-363)
- `include/ck_tile/ops/flatmm/pipeline/mx_flatmm_pipeline_agmem_bgmem_creg_v1_policy.hpp`  (+233/-74)
- `include/ck_tile/core/tensor/tile_window.hpp`  (+75/-21)
- `example/ck_tile/18_flatmm/mxgemm/mx_flatmm.hpp`  (+84/-1)
- `example/ck_tile/18_flatmm/mxgemm/mx_flatmm.cpp`  (+37/-27)
- `example/ck_tile/18_flatmm/mxgemm/mxfp4_flatmm.hpp`  (+0/-60)
- `include/ck_tile/core/tensor/load_tile.hpp`  (+24/-20)
- `example/ck_tile/18_flatmm/mxgemm/mx_flatmm_instance.cmake`  (+17/-12)
- `include/ck_tile/ops/flatmm/kernel/mx_flatmm_kernel.hpp`  (+18/-10)
- `include/ck_tile/host/reference/reference_gemm.hpp`  (+6/-0)
- `example/ck_tile/18_flatmm/mxgemm/run_mx_flatmm.inc`  (+1/-4)
- `example/ck_tile/18_flatmm/CMakeLists.txt`  (+3/-1)
- `include/ck_tile/core/tensor/tensor_view.hpp`  (+2/-2)
- `include/ck_tile/ops/flatmm/pipeline/flatmm_pipeline_agmem_bgmem_creg_v1.hpp`  (+3/-0)
- `include/ck_tile/core/numeric/integral_constant.hpp`  (+2/-0)

## Key added lines (kernel files)

**`example/ck_tile/18_flatmm/mxgemm/mx_flatmm.cpp`**
```
std::cout << "Run " << ck_tile::gemm_prec_str<ADataType, BDataType>() << " Flatmm kernel " //
template <ck_tile::index_t N_Warp_Tile, typename dtype>
auto preShuffleWeight(ck_tile::HostTensor<dtype>& src)
auto src_lengths          = src.get_lengths();
```

**`example/ck_tile/18_flatmm/mxgemm/mx_flatmm.hpp`**
```
struct MXfp4_FlatmmConfig16
static constexpr ck_tile::index_t M_Tile = 128;
static constexpr ck_tile::index_t N_Tile = 512;
static constexpr ck_tile::index_t K_Tile = 256;
```

**`example/ck_tile/18_flatmm/mxgemm/run_mx_flatmm.inc`**
```
const auto b_shuffled_host  = preShuffleWeight<FlatmmConfig::N_Warp_Tile>(b_origin_host);
```

**`include/ck_tile/core/numeric/integral_constant.hpp`**
```
using true_type     = bool_constant<true>;
using false_type    = bool_constant<false>;
```

**`include/ck_tile/core/tensor/load_tile.hpp`**
```
typename offset_t,
typename = std::enable_if_t<std::is_class_v<TileWindow_>>>
offset_t offset,
typename offset_t,
```
