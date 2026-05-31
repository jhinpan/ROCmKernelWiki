# Diff summary

- **files changed:** 14
- **lines:** +205 / -119
- **kernel-ish files:** 14

## Files (by churn)

- `include/ck_tile/ops/gemm/kernel/universal_gemm_kernel.hpp`  (+87/-37)
- `include/ck_tile/ops/epilogue/cshuffle_epilogue.hpp`  (+26/-9)
- `test/ck_tile/gemm/test_gemm_pipeline_ut_cases.inc`  (+17/-14)
- `include/ck_tile/core/tensor/transpose_tile.hpp`  (+3/-26)
- `include/ck_tile/core/arch/arch.hpp`  (+8/-18)
- `include/ck_tile/ops/gemm/warp/warp_gemm_attribute_wmma.hpp`  (+15/-2)
- `include/ck_tile/ops/gemm/pipeline/gemm_universal_pipeline_ag_bg_cr_policy.hpp`  (+7/-6)
- `include/ck_tile/ops/gemm/warp/warp_gemm.hpp`  (+10/-0)
- `include/ck_tile/ops/gemm/warp/warp_gemm_attribute_wmma_impl_8bit_traits.hpp`  (+10/-0)
- `include/ck_tile/ops/gemm/pipeline/tile_gemm_traits.hpp`  (+4/-4)
- `include/ck_tile/ops/gemm/warp/warp_gemm_attribute_wmma_impl_16bit_traits.hpp`  (+8/-0)
- `include/ck_tile/ops/gemm/warp/warp_gemm_attribute_wmma_impl.hpp`  (+4/-3)
- `include/ck_tile/ops/gemm/warp/warp_gemm_attribute_wmma_impl_base_traits.hpp`  (+4/-0)
- `include/ck_tile/ops/gemm/warp/warp_gemm_dispatcher.hpp`  (+2/-0)

## Key added lines (kernel files)

**`include/ck_tile/core/arch/arch.hpp`**
```
return gfx103_t{};
return gfx950_t{};
return gfx9_t{};
return detail::get_n_lds_banks(get_device_arch());
```

**`include/ck_tile/core/tensor/transpose_tile.hpp`**
```
constexpr index_t NDimY = InTensor::get_tile_distribution().get_num_of_dimension_y();
static_for<0, NDimY, 1>{}([&](auto i) { y_dim_out_to_in_(i) = NDimY - 1 - i; });
constexpr index_t y_dim_vec_out = 0;
```

**`include/ck_tile/ops/epilogue/cshuffle_epilogue.hpp`**
```
if constexpr(is_any_of<ADataType, pk_int4_t, pk_fp4_t>::value ||
is_any_of<BDataType, pk_int4_t, pk_fp4_t>::value)
return tile_distribution_encoding<
sequence<>,
```

**`include/ck_tile/ops/gemm/kernel/universal_gemm_kernel.hpp`**
```
bool AsTensorIsValid   = {true};
AsTensorIsValid = false;
const auto remainder = kargs.K % vectorSizeA;
constexpr ck_tile::index_t APackedSize =
```

**`include/ck_tile/ops/gemm/pipeline/gemm_universal_pipeline_ag_bg_cr_policy.hpp`**
```
using ADataType                 = remove_cvref_t<typename Problem::ADataType>;
constexpr auto a_lds_block_desc = Derived::template MakeALdsBlockDescriptor<Problem>();
constexpr index_t smem_size_a   = integer_least_multiple(
a_lds_block_desc.get_element_space_size() * sizeof(ADataType), 16);
```
