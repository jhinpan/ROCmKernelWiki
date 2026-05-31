# Diff summary

- **files changed:** 13
- **lines:** +1231 / -187
- **kernel-ish files:** 11

## Files (by churn)

- `include/ck_tile/ops/gemm/pipeline/wp_pipeline_agmem_bgmem_creg_v2.hpp`  (+1070/-0)
- `include/ck_tile/ops/gemm/pipeline/wp_pipeline_agmem_bgmem_creg_base_policy.hpp`  (+14/-163)
- `script/gemm_profile.sh`  (+107/-0)
- `example/ck_tile/03_gemm/gemm_utils.hpp`  (+22/-11)
- `include/ck_tile/ops/gemm/pipeline/wp_pipeline_agmem_bgmem_creg_v1.hpp`  (+6/-8)
- `example/ck_tile/03_gemm/CMakeLists.txt`  (+6/-0)
- `example/ck_tile/03_gemm/gemm_weight_preshuffle.cpp`  (+1/-3)
- `include/ck_tile/ops/gemm.hpp`  (+2/-1)
- `include/ck_tile/ops/gemm/kernel/gemm_tile_partitioner.hpp`  (+1/-1)
- `example/ck_tile/18_flatmm/flatmm_basic.cpp`  (+1/-0)
- `include/ck_tile/ops/flatmm/pipeline/flatmm_pipeline_agmem_bgmem_creg_v1.hpp`  (+1/-0)
- `include/ck_tile/ops/flatmm/kernel/flatmm_kernel.hpp`  (+0/-0)
- `include/ck_tile/ops/gemm/kernel/gemm_kernel.hpp`  (+0/-0)

## Key added lines (kernel files)

**`example/ck_tile/03_gemm/gemm_utils.hpp`**
```
struct GemmConfigPreshuffle_1 : public GemmConfigBase
static constexpr ck_tile::index_t Pipeline = CK_TILE_PIPELINE_PRESHUFFLE_V1;
struct GemmConfigPreshuffle_2 : public GemmConfigBase
static constexpr ck_tile::index_t M_Warp_Tile = 16;
```

**`example/ck_tile/03_gemm/gemm_weight_preshuffle.cpp`**
```
return !run_gemm_example<GemmConfigPreshuffle_2>(argc, argv);
```

**`include/ck_tile/ops/gemm/kernel/gemm_tile_partitioner.hpp`**
```
CK_TILE_HOST_DEVICE static auto
```

**`include/ck_tile/ops/gemm/pipeline/wp_pipeline_agmem_bgmem_creg_base_policy.hpp`**
```
: public UniversalGemmBasePolicy<UniversalWeightPreshufflePipelineAgBgCrPolicy>
using BasePolicy = UniversalGemmBasePolicy<UniversalWeightPreshufflePipelineAgBgCrPolicy>;
using ADataType              = remove_cvref_t<typename Problem::ADataType>;
number<kMPerBlock / MLdsLayer>{},
```

**`include/ck_tile/ops/gemm/pipeline/wp_pipeline_agmem_bgmem_creg_v1.hpp`**
```
auto b_flat_dram_window =
make_tile_window(b_flat_dram_block_window_tmp.get_bottom_tensor_view(),
make_tuple(number<flatNPerWarp>{}, number<flatKPerWarp>{}),
b_flat_dram_block_window_tmp.get_window_origin(),
```
