# Diff summary

- **files changed:** 18
- **lines:** +433 / -15
- **kernel-ish files:** 18

## Files (by churn)

- `experimental/builder/include/ck_tile/builder/reflect/instance_traits_tile_grouped_convolution_forward.hpp`  (+140/-0)
- `experimental/builder/test/test_fwd_instance_traits.cpp`  (+123/-0)
- `experimental/builder/include/ck_tile/builder/reflect/instance_traits_util.hpp`  (+76/-5)
- `include/ck_tile/ops/grouped_convolution/kernel/grouped_convolution_forward_kernel.hpp`  (+17/-0)
- `include/ck_tile/core/arch/arch.hpp`  (+5/-5)
- `include/ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_comp_async.hpp`  (+7/-0)
- `include/ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_comp_v3.hpp`  (+7/-0)
- `include/ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_comp_v4.hpp`  (+7/-0)
- `include/ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_comp_v5.hpp`  (+7/-0)
- `include/ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_comp_v6.hpp`  (+7/-0)
- `include/ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_mem.hpp`  (+7/-0)
- `include/ck_tile/ops/gemm/pipeline/gemm_pipeline_agmem_bgmem_creg_v1.hpp`  (+7/-0)
- `include/ck_tile/ops/gemm/pipeline/gemm_pipeline_agmem_bgmem_creg_v2.hpp`  (+7/-0)
- `include/ck_tile/ops/gemm/pipeline/wp_pipeline_agmem_bgmem_creg_v2.hpp`  (+7/-0)
- `include/ck_tile/ops/gemm/kernel/grouped_gemm_kernel.hpp`  (+3/-3)

## Key added lines (kernel files)

**`experimental/builder/include/ck_tile/builder/reflect/instance_traits_tile_grouped_convolution_forward.hpp`**
```
namespace ck_tile::device {
template <typename GroupedConvTraitsType_,
typename TilePartitioner_,
typename GemmPipeline_,
```

**`experimental/builder/include/ck_tile/builder/reflect/instance_traits_util.hpp`**
```
if constexpr(std::is_same_v<T, ck::half_t> || std::is_same_v<T, ck_tile::half_t>)
else if constexpr(std::is_same_v<T, ck::bhalf_t> || std::is_same_v<T, ck_tile::bf16_t>)
else if constexpr(std::is_same_v<T, ck::f8_t> || std::is_same_v<T, ck_tile::fp8_t>)
else if constexpr(std::is_same_v<T, ck::bf8_t> || std::is_same_v<T, ck_tile::bf8_t>)
```

**`experimental/builder/test/test_fwd_instance_traits.cpp`**
```
TEST(InstanceTraits, TileInstanceStringReturnsCorrectFormat)
using GroupedConvTraitsType =
ck_tile::GroupedConvTraits<2 /*NDimSpatial*/,
ck_tile::ConvolutionSpecialization::Default /*ConvSpec*/,
```

**`include/ck_tile/core/arch/arch.hpp`**
```
__attribute__((address_space(      \
__device__ T* cast_pointer_to_generic_address_space(T CK_TILE_CONSTANT_ADDRESS_SPACE* p)
__host__ __device__ T CK_TILE_CONSTANT_ADDRESS_SPACE* cast_pointer_to_constant_address_space(T* p)
return (T CK_TILE_CONSTANT_ADDRESS_SPACE*)p; // NOLINT(old-style-cast)
```

**`include/ck_tile/ops/gemm/kernel/grouped_gemm_kernel.hpp`**
```
using ConstantPointer = const void CK_TILE_CONSTANT_ADDRESS_SPACE*;
CK_TILE_DEVICE void operator()(const void CK_TILE_CONSTANT_ADDRESS_SPACE* gemm_descs_const,
CK_TILE_DEVICE void operator()(const void CK_TILE_CONSTANT_ADDRESS_SPACE* gemm_descs_const,
```
