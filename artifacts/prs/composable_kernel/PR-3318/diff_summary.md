# Diff summary

- **files changed:** 10
- **lines:** +222 / -155
- **kernel-ish files:** 9

## Files (by churn)

- `test/ck_tile/gemm_block_scale/test_gemm_quant_typed.cpp`  (+90/-82)
- `include/ck_tile/ops/gemm_quant/kernel/gemm_quant_kernel.hpp`  (+33/-16)
- `test/ck_tile/gemm_block_scale/test_gemm_quant_base.hpp`  (+26/-23)
- `example/ck_tile/38_block_scale_gemm/run_gemm_quant_example.inc`  (+27/-16)
- `include/ck_tile/ops/gemm_quant/pipeline/gemm_group_quant_utils.hpp`  (+21/-0)
- `include/ck_tile/ops/gemm_quant/pipeline/gemm_aquant_pipeline_ag_bg_cr_policy.hpp`  (+14/-4)
- `include/ck_tile/ops/gemm_quant/pipeline/gemm_aquant_pipeline_ag_bg_cr_v3.hpp`  (+5/-6)
- `test/ck_tile/gemm_block_scale/test_gemm_quant_fixtures.hpp`  (+5/-4)
- `include/ck_tile/ops/gemm_quant/pipeline/gemm_aquant_pipeline_ag_bg_cr_base.hpp`  (+0/-4)
- `CHANGELOG.md`  (+1/-0)

## Key added lines (kernel files)

**`example/ck_tile/38_block_scale_gemm/run_gemm_quant_example.inc`**
```
typename AQLayout,
typename BQLayout,
using GemmTraits = ck_tile::TileGemmQuantTraits<GemmConfig::kPadM,
GemmConfig::kPadN,
```

**`include/ck_tile/ops/gemm_quant/kernel/gemm_quant_kernel.hpp`**
```
if constexpr(std::is_same_v<AQLayout, tensor_layout::gemm::RowMajor>)
return make_naive_tensor_view<address_space_enum::global>(
make_tuple(kargs.M, kargs.QK_A),
make_tuple(kargs.stride_AQ, 1),
```

**`include/ck_tile/ops/gemm_quant/pipeline/gemm_aquant_pipeline_ag_bg_cr_policy.hpp`**
```
else if constexpr(std::is_same_v<AQLayout, tensor_layout::gemm::RowMajor>)
using TileEncodingPattern =
tile_distribution_encoding_pattern_aq<BlockGemmShape,
WarpGemm,
```

**`include/ck_tile/ops/gemm_quant/pipeline/gemm_aquant_pipeline_ag_bg_cr_v3.hpp`**
```
PreshuffleQuant
? make_array(ck_tile::integer_least_multiple(m, MPerBlock) /
BlockGemm::WarpGemm::kM,
: (is_aq_col_major ? make_array(KPerBlockAQ, 0) : make_array(0, KPerBlockAQ));
```

**`include/ck_tile/ops/gemm_quant/pipeline/gemm_group_quant_utils.hpp`**
```
CK_TILE_HOST_DEVICE static constexpr auto make_2d_static_tile_distribution_transposed()
constexpr index_t Y0 = YPerTile;
constexpr index_t X0 = 1;
constexpr index_t X1 = MIterPerWarp ? MIterPerWarp : 1;
```
