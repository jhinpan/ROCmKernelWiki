# Diff summary

- **files changed:** 7
- **lines:** +109 / -15
- **kernel-ish files:** 6

## Files (by churn)

- `test/ck_tile/gemm_block_scale/test_gemm_quant_fixtures.hpp`  (+48/-4)
- `include/ck_tile/ops/gemm_quant/block/block_universal_gemm_as_aquant_bs_cr.hpp`  (+34/-7)
- `test/ck_tile/gemm_block_scale/test_gemm_quant_typed.cpp`  (+20/-1)
- `test/ck_tile/gemm_block_scale/CMakeLists.txt`  (+3/-1)
- `include/ck_tile/ops/gemm_quant/pipeline/tile_gemm_quant_traits.hpp`  (+2/-1)
- `include/ck_tile/ops/gemm_quant/kernel/gemm_quant_kernel.hpp`  (+1/-1)
- `test/ck_tile/gemm_block_scale/test_gemm_quant_base.hpp`  (+1/-0)

## Key added lines (kernel files)

**`include/ck_tile/ops/gemm_quant/block/block_universal_gemm_as_aquant_bs_cr.hpp`**
```
constexpr auto tbuf_offset = number<
typename CBlockTensor::ThreadTensorDesc{}.calculate_offset(
merge_sequences(sequence<mIter, nIter>{},
c_warp_y_index_zeros)) /
```

**`include/ck_tile/ops/gemm_quant/kernel/gemm_quant_kernel.hpp`**
```
struct is_quantpreshuffle_enabled<T, std::void_t<decltype(T::PreshuffleQuant)>>
```

**`include/ck_tile/ops/gemm_quant/pipeline/tile_gemm_quant_traits.hpp`**
```
bool TransposeC_          = false,
static constexpr bool TransposeC            = TransposeC_;
```

**`test/ck_tile/gemm_block_scale/test_gemm_quant_base.hpp`**
```
GemmConfig::TransposeC,
```

**`test/ck_tile/gemm_block_scale/test_gemm_quant_fixtures.hpp`**
```
struct GemmConfigPreshuffleQuant : public GemmConfigBase
static constexpr bool PreshuffleQuant = true;
struct GemmConfigTransposeC : public GemmConfigBase
static constexpr bool TransposeC = true;
```
