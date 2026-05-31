# Diff summary

- **files changed:** 12
- **lines:** +263 / -1066
- **kernel-ish files:** 12

## Files (by churn)

- `test/ck_tile/gemm_block_scale/test_gemm_aquant_utils.hpp`  (+24/-462)
- `example/ck_tile/38_block_scale_gemm/gemm_utils.hpp`  (+16/-404)
- `test/ck_tile/gemm_block_scale/test_run_gemm_aquant_example.inc`  (+58/-39)
- `example/ck_tile/38_block_scale_gemm/gemm_aquant_preshuffle.cpp`  (+28/-35)
- `example/ck_tile/38_block_scale_gemm/gemm_aquant_basic.cpp`  (+25/-35)
- `include/ck_tile/ops/gemm_group_quant/kernel/gemm_aquant_kernel.hpp`  (+29/-29)
- `example/ck_tile/38_block_scale_gemm/run_gemm_aquant_example.inc`  (+24/-24)
- `include/ck_tile/ops/gemm_group_quant/pipeline/gemm_aquant_pipeline_ag_bg_cr_policy.hpp`  (+17/-17)
- `include/ck_tile/ops/gemm_group_quant/block/block_universal_gemm_as_aquant_bs_cr.hpp`  (+23/-8)
- `include/ck_tile/ops/gemm_group_quant/pipeline/gemm_aquant_pipeline_ag_bg_cr_v3.hpp`  (+10/-4)
- `include/ck_tile/ops/gemm_group_quant/pipeline/gemm_group_quant_utils.hpp`  (+7/-7)
- `include/ck_tile/ops/gemm_group_quant/pipeline/tile_gemm_aquant_traits.hpp`  (+2/-2)

## Key added lines (kernel files)

**`example/ck_tile/38_block_scale_gemm/gemm_aquant_basic.cpp`**
```
template <typename GemmConfig,
typename ADataType,
uint32_t QuantGroupSize>
constexpr ck_tile::index_t M_Tile = GemmConfig::M_Tile;
```

**`example/ck_tile/38_block_scale_gemm/gemm_aquant_preshuffle.cpp`**
```
template <typename GemmConfig,
typename ADataType,
uint32_t QuantGroupSize>
constexpr ck_tile::index_t M_Tile = GemmConfig::M_Tile;
```

**`example/ck_tile/38_block_scale_gemm/gemm_utils.hpp`**
```
static constexpr bool PreshuffleQuant           = false;
static constexpr bool DoubleSmemBuffer          = false;
struct GemmConfigDecode : public GemmConfigBase
static constexpr ck_tile::index_t M_Tile = 16;
```

**`example/ck_tile/38_block_scale_gemm/run_gemm_aquant_example.inc`**
```
template <typename GemmConfig,
typename ADataType,
uint32_t QuantGroupSize>
float ave_time = gemm_calc_aquant<GemmConfig,
```

**`include/ck_tile/ops/gemm_group_quant/block/block_universal_gemm_as_aquant_bs_cr.hpp`**
```
static constexpr bool PreshuffleQuant = Problem::Traits::PreshuffleQuant;
if constexpr(Traits::PreshuffleQuant)
decltype(threadIdx.x) pull_from_lane = 0;
if constexpr(WarpGemm::kM == 16)
```
