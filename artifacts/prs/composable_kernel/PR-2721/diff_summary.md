# Diff summary

- **files changed:** 13
- **lines:** +808 / -178
- **kernel-ish files:** 9

## Files (by churn)

- `example/ck_tile/17_grouped_gemm/grouped_gemm_preshuffle.cpp`  (+234/-0)
- `example/ck_tile/17_grouped_gemm/grouped_gemm.cpp`  (+203/-7)
- `example/ck_tile/17_grouped_gemm/run_grouped_gemm_example.inc`  (+47/-115)
- `example/ck_tile/17_grouped_gemm/grouped_gemm.hpp`  (+97/-4)
- `example/ck_tile/17_grouped_gemm/README.md`  (+58/-20)
- `include/ck_tile/ops/gemm/pipeline/wp_pipeline_agmem_bgmem_creg_base_policy.hpp`  (+67/-0)
- `include/ck_tile/ops/gemm/kernel/grouped_gemm_kernel.hpp`  (+42/-14)
- `include/ck_tile/ops/gemm/pipeline/wp_pipeline_agmem_bgmem_creg_v2.hpp`  (+46/-7)
- `include/ck_tile/ops/gemm/block/block_wp_asmem_bsmem_creg_v1.hpp`  (+3/-7)
- `script/gemm_profile.sh`  (+8/-2)
- `CHANGELOG.md`  (+1/-1)
- `include/ck_tile/ops/gemm/pipeline/tile_gemm_traits.hpp`  (+1/-1)
- `example/ck_tile/17_grouped_gemm/CMakeLists.txt`  (+1/-0)

## Key added lines (kernel files)

**`example/ck_tile/17_grouped_gemm/grouped_gemm.cpp`**
```
template <typename GemmConfig,
typename ADataType,
typename BDataType,
typename DsDataType,
```

**`example/ck_tile/17_grouped_gemm/grouped_gemm.hpp`**
```
template <typename PrecType, ck_tile::index_t M_Warp_Tile>
constexpr ck_tile::index_t get_k_warp_tile_flatmm()
if constexpr(M_Warp_Tile == 32)
return sizeof(PrecType) == 2 ? 16 : 64;
```

**`example/ck_tile/17_grouped_gemm/grouped_gemm_preshuffle.cpp`**
```
template <typename GemmConfig,
typename ADataType,
typename BDataType,
typename DsDataType,
```

**`example/ck_tile/17_grouped_gemm/run_grouped_gemm_example.inc`**
```
if constexpr(!GemmConfig::Persistent)
ave_time = grouped_gemm<GemmConfig,
ADataType,
if(GemmConfig::Preshuffle)
```

**`include/ck_tile/ops/gemm/block/block_wp_asmem_bsmem_creg_v1.hpp`**
```
static constexpr auto config = BlockPolicy::template GetWarpGemmMWarpNWarp<Problem>();
using WG                     = remove_cvref_t<decltype(config.template at<0>())>;
```
