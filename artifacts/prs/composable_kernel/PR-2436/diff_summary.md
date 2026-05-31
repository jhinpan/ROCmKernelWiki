# Diff summary

- **files changed:** 26
- **lines:** +1165 / -580
- **kernel-ish files:** 25

## Files (by churn)

- `include/ck_tile/core/tensor/load_tile_transpose.hpp`  (+206/-122)
- `include/ck_tile/ops/gemm/warp/warp_gemm_attribute_mfma.hpp`  (+154/-125)
- `include/ck_tile/core/utility/debug.hpp`  (+156/-0)
- `include/ck_tile/ops/gemm/block/block_universal_gemm_as_bs_cr.hpp`  (+119/-35)
- `include/ck_tile/ops/gemm/pipeline/gemm_universal_pipeline_ag_bg_cr_policy.hpp`  (+96/-47)
- `include/ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_comp_v4.hpp`  (+76/-65)
- `include/ck_tile/ops/gemm/warp/warp_gemm.hpp`  (+60/-22)
- `include/ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_mem.hpp`  (+51/-28)
- `include/ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_base.hpp`  (+50/-17)
- `include/ck_tile/core/tensor/buffer_view.hpp`  (+25/-40)
- `include/ck_tile/ops/gemm/warp/warp_gemm_dispatcher.hpp`  (+44/-17)
- `include/ck_tile/core/arch/amd_transpose_load_encoding.hpp`  (+30/-28)
- `include/ck_tile/ops/gemm/block/block_gemm_asmem_bsmem_creg_v1_default_policy.hpp`  (+38/-4)
- `include/ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_comp_v3.hpp`  (+13/-7)
- `include/ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_comp_v4_default_policy.hpp`  (+15/-3)

## Key added lines (kernel files)

**`example/ck_tile/37_transpose/transpose_policy.hpp`**
```
typename OutputTileDistributionTraits<typename decltype(input_dstr)::DstrEncode,
typename Problem::DataType>::TransposedDstrEncode;
constexpr index_t kLaneGroupSize      = 16;
kLaneGroupSize,
```

**`include/ck_tile/core/arch/amd_buffer_addressing.hpp`**
```
else if constexpr(std::is_same_v<remove_cvref_t<T>, ck_tile::fp8_t> ||
std::is_same_v<remove_cvref_t<T>, ck_tile::bf8_t> ||
std::is_same_v<remove_cvref_t<T>, ck_tile::int8_t>)
typedef __attribute__((__vector_size__(2 * sizeof(index_t)))) index_t llvm_i32x2_t;
```

**`include/ck_tile/core/arch/amd_buffer_addressing_builtins.hpp`**
```
else if constexpr(std::is_same_v<remove_cvref_t<T>, ck_tile::fp8_t> ||
std::is_same_v<remove_cvref_t<T>, ck_tile::bf8_t> ||
std::is_same_v<remove_cvref_t<T>, ck_tile::int8_t>)
typedef __attribute__((__vector_size__(2 * sizeof(index_t)))) index_t llvm_i32x2_t;
```

**`include/ck_tile/core/arch/amd_transpose_load_encoding.hpp`**
```
template <typename T, index_t LaneGroupSize = 16, typename = void>
template <typename T, index_t LaneGroupSize>
struct LaneGroupTransposeTraits<T, LaneGroupSize, std::enable_if_t<sizeof(T) == 2>>
static_assert(LaneGroupSize == 16 || LaneGroupSize == 32 || LaneGroupSize == 64,
```

**`include/ck_tile/core/tensor/buffer_view.hpp`**
```
(std::is_same_v<remove_cvref_t<T>, int8_t> && std::is_same_v<remove_cvref_t<X>, int8_t>) ||
(std::is_same_v<remove_cvref_t<T>, int8_t> && std::is_same_v<remove_cvref_t<X>, int8x2_t>) ||
(std::is_same_v<remove_cvref_t<T>, int8_t> && std::is_same_v<remove_cvref_t<X>, int8x4_t>) ||
(std::is_same_v<remove_cvref_t<T>, int8_t> && std::is_same_v<remove_cvref_t<X>, int8x8_t>) ||
```
