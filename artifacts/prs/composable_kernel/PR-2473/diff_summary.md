# Diff summary

- **files changed:** 35
- **lines:** +4864 / -13
- **kernel-ish files:** 30

## Files (by churn)

- `test/ck_tile/gemm_block_scale/test_gemm_aquant_utils.hpp`  (+681/-0)
- `include/ck_tile/ops/gemm_group_quant/kernel/gemm_aquant_kernel.hpp`  (+679/-0)
- `example/ck_tile/38_block_scale_gemm/gemm_utils.hpp`  (+675/-0)
- `test/ck_tile/gemm_block_scale/test_run_gemm_aquant_example.inc`  (+577/-0)
- `include/ck_tile/ops/gemm_group_quant/block/block_universal_gemm_as_aquant_bs_cr.hpp`  (+489/-0)
- `include/ck_tile/ops/gemm_group_quant/pipeline/gemm_aquant_pipeline_ag_bg_cr_v3.hpp`  (+476/-0)
- `example/ck_tile/38_block_scale_gemm/run_gemm_aquant_example.inc`  (+259/-0)
- `example/ck_tile/38_block_scale_gemm/gemm_aquant_basic.cpp`  (+226/-0)
- `include/ck_tile/ops/gemm_group_quant/pipeline/gemm_aquant_pipeline_problem.hpp`  (+121/-0)
- `include/ck_tile/host/reference/reference_gemm.hpp`  (+104/-0)
- `include/ck_tile/ops/gemm_group_quant/pipeline/gemm_group_quant_utils.hpp`  (+95/-0)
- `include/ck_tile/ops/gemm_group_quant/pipeline/gemm_aquant_pipeline_ag_bg_cr_policy.hpp`  (+93/-0)
- `include/ck_tile/ops/elementwise/unary_element_wise_operation.hpp`  (+90/-0)
- `include/ck_tile/host/fill.hpp`  (+55/-0)
- `include/ck_tile/ops/gemm_group_quant/pipeline/gemm_aquant_pipeline_ag_bg_cr_base.hpp`  (+53/-0)

## Key added lines (kernel files)

**`example/ck_tile/03_gemm/gemm_utils.hpp`**
```
static constexpr ck_tile::index_t M_Tile = 16;
static constexpr ck_tile::index_t N_Tile = 64;
static constexpr ck_tile::index_t K_Tile = 256 / sizeof(PrecType);
static constexpr ck_tile::index_t M_Warp = 1;
```

**`example/ck_tile/38_block_scale_gemm/gemm_aquant_basic.cpp`**
```
template <typename ADataType,
typename AQDataType,
typename BDataType,
typename AccDataType,
```

**`example/ck_tile/38_block_scale_gemm/gemm_utils.hpp`**
```
template <typename PrecType, ck_tile::index_t M_Warp_Tile>
constexpr ck_tile::index_t get_k_warp_tile()
constexpr bool is_8bit_float =
std::is_same_v<PrecType, ck_tile::fp8_t> || std::is_same_v<PrecType, ck_tile::bf8_t>;
```

**`example/ck_tile/38_block_scale_gemm/run_gemm_aquant_example.inc`**
```
template <typename Layout>
static constexpr inline auto is_row_major(Layout layout_)
return ck_tile::bool_constant<std::is_same_v<ck_tile::remove_cvref_t<decltype(layout_)>,
ck_tile::tensor_layout::gemm::RowMajor>>{};
```

**`include/ck_tile/core/numeric/pk_int4.hpp`**
```
CK_TILE_HOST_DEVICE fp32x2_t pk_int4_t_to_fp32x2_t_signed_conversion(const pk_int4_t& x)
uint8_t x_u8 = ck_tile::bit_cast<uint8_t>(x);
float x_l = ((x_u8 & 0x0f) >> 0);
float x_h = ((x_u8 & 0xf0) >> 4);
```
