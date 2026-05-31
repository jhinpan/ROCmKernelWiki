# Diff summary

- **files changed:** 7
- **lines:** +163 / -53
- **kernel-ish files:** 7

## Files (by churn)

- `include/ck_tile/ops/gemm_quant/block/block_universal_gemm_as_aquant_bs_bquant_cr.hpp`  (+57/-28)
- `include/ck_tile/ops/gemm_quant/block/block_universal_gemm_ar_aquant_flatbr_bquant_cr.hpp`  (+40/-4)
- `example/ck_tile/38_block_scale_gemm/gemm_utils.hpp`  (+29/-0)
- `include/ck_tile/ops/gemm_quant/pipeline/gemm_wp_abquant_pipeline_ag_bg_cr_v2.hpp`  (+18/-8)
- `example/ck_tile/38_block_scale_gemm/gemm_abquant_quantgrouped.cpp`  (+11/-5)
- `include/ck_tile/core/tensor/sweep_tile.hpp`  (+6/-6)
- `example/ck_tile/38_block_scale_gemm/run_gemm_quant_example.inc`  (+2/-2)

## Key added lines (kernel files)

**`example/ck_tile/38_block_scale_gemm/gemm_abquant_quantgrouped.cpp`**
```
using GemmConfig = GemmConfigABQuantPrefill<T>;
template <typename T>
using GemmConfigPreshuffleB = GemmConfigPreshuffleB_ABQuant_Prefill<T>;
return run_gemm_example_prec_type<GemmConfigPreshuffleB<ck_tile::fp8_t>,
```

**`example/ck_tile/38_block_scale_gemm/gemm_utils.hpp`**
```
template <typename PrecType>
struct GemmConfigPreshuffleB_ABQuant_Prefill : public GemmConfigPreshuffleB_BQuant_Prefill<PrecType>
static constexpr ck_tile::index_t M_Warp = 2;
static constexpr ck_tile::index_t N_Warp = 2;
```

**`example/ck_tile/38_block_scale_gemm/run_gemm_quant_example.inc`**
```
constexpr bool transpose_c = QuantMode == ck_tile::QuantType::ABQuantGrouped;
transpose_c,
```

**`include/ck_tile/core/tensor/sweep_tile.hpp`**
```
using DstrSpanImpl = typename remove_cvref_t<TileDistributedSpan_>::Impl;
if constexpr(DstrSpanImpl::size() == 0) // handle the 0-dim span case
f(detail::make_tile_distributed_index(sequence<>{}));
static_ford<DstrSpanImpl>{}(
```

**`include/ck_tile/ops/gemm_quant/block/block_universal_gemm_ar_aquant_flatbr_bquant_cr.hpp`**
```
auto q_block_tensor = aq_block_tensor;
constexpr bool SimpleDequant =
Traits::NQPerBlock == 1 &&
AccTensor::get_distributed_spans()[I0].impl_.size() == 0; // c_transpose
```
