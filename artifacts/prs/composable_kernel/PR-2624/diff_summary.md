# Diff summary

- **files changed:** 13
- **lines:** +682 / -243
- **kernel-ish files:** 12

## Files (by churn)

- `example/ck_tile/38_block_scale_gemm/gemm_aquant_preshuffle.cpp`  (+238/-0)
- `include/ck_tile/ops/gemm_group_quant/block/block_universal_gemm_as_aquant_bs_cr.hpp`  (+150/-76)
- `include/ck_tile/ops/gemm_group_quant/kernel/gemm_aquant_kernel.hpp`  (+101/-27)
- `example/ck_tile/38_block_scale_gemm/gemm_utils.hpp`  (+34/-69)
- `example/ck_tile/38_block_scale_gemm/run_gemm_aquant_example.inc`  (+52/-19)
- `include/ck_tile/ops/gemm_group_quant/pipeline/gemm_group_quant_utils.hpp`  (+41/-20)
- `include/ck_tile/ops/gemm_group_quant/pipeline/gemm_aquant_pipeline_ag_bg_cr_policy.hpp`  (+29/-8)
- `example/ck_tile/38_block_scale_gemm/gemm_aquant_basic.cpp`  (+19/-11)
- `test/ck_tile/gemm_block_scale/test_run_gemm_aquant_example.inc`  (+7/-4)
- `include/ck_tile/ops/gemm_group_quant/pipeline/gemm_aquant_pipeline_ag_bg_cr_v3.hpp`  (+5/-5)
- `include/ck_tile/ops/gemm_group_quant/pipeline/gemm_aquant_pipeline_ag_bg_cr_base.hpp`  (+1/-4)
- `example/ck_tile/38_block_scale_gemm/CMakeLists.txt`  (+3/-0)
- `include/ck_tile/ops/gemm_group_quant/pipeline/tile_gemm_aquant_traits.hpp`  (+2/-0)

## Key added lines (kernel files)

**`example/ck_tile/38_block_scale_gemm/gemm_aquant_basic.cpp`**
```
uint32_t QuantGroupSize,
bool Preshuffle = false>
ck_tile::TileGemmAQuantTraits<kPadM, kPadN, kPadK, Preshuffle, ALayout, BLayout, CLayout>;
template <typename GemmConfig, typename TypeConfig, uint32_t QuantGroupSize>
```

**`example/ck_tile/38_block_scale_gemm/gemm_aquant_preshuffle.cpp`**
```
template <typename ADataType,
typename AQDataType,
typename BDataType,
typename AccDataType,
```

**`example/ck_tile/38_block_scale_gemm/gemm_utils.hpp`**
```
constexpr ck_tile::index_t get_k_from_preshuffled_warp_tile()
static constexpr ck_tile::index_t K_Tile = 256 / sizeof(PrecType);
static constexpr ck_tile::index_t K_Warp_Tile =
get_k_from_preshuffled_warp_tile<PrecType, M_Warp_Tile>();
```

**`example/ck_tile/38_block_scale_gemm/run_gemm_aquant_example.inc`**
```
template <typename T>
auto shuffle_aq(const ck_tile::HostTensor<T>& t, int block_aq_k)
if(t.get_lengths().size() != 2)
throw std::runtime_error("Host tensor is not rank 2 tensor.");
```

**`include/ck_tile/ops/gemm_group_quant/block/block_universal_gemm_as_aquant_bs_cr.hpp`**
```
static constexpr bool Preshuffle = Problem::Traits::Preshuffle;
constexpr auto warp_size = get_warp_size();
if constexpr(Traits::Preshuffle)
constexpr auto tbuf_offset =
```
