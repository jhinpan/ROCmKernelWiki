# Diff summary

- **files changed:** 8
- **lines:** +143 / -10
- **kernel-ish files:** 7

## Files (by churn)

- `include/ck_tile/ops/gemm/warp/warp_gemm_attribute_mfma_impl.hpp`  (+98/-0)
- `include/ck_tile/ops/epilogue/default_2d_epilogue.hpp`  (+15/-6)
- `include/ck_tile/ops/gemm/warp/warp_gemm.hpp`  (+12/-0)
- `tile_engine/ops/gemm/CMakeLists.txt`  (+7/-1)
- `include/ck_tile/ops/gemm/warp/warp_gemm_dispatcher.hpp`  (+5/-0)
- `tile_engine/ops/gemm/gemm_instance_builder.py`  (+3/-1)
- `include/ck_tile/ops/epilogue/cshuffle_epilogue.hpp`  (+2/-1)
- `include/ck_tile/ops/gemm/block/block_universal_gemm_as_bs_cr.hpp`  (+1/-1)

## Key added lines (kernel files)

**`include/ck_tile/ops/epilogue/cshuffle_epilogue.hpp`**
```
std::conditional_t<std::is_same_v<BDataType, pk_int4_t>, ADataType, BDataType>;
```

**`include/ck_tile/ops/epilogue/default_2d_epilogue.hpp`**
```
template <typename ADataType_,
typename BDataType_,
typename AccDataType_,
using ADataType                        = remove_cvref_t<ADataType_>;
```

**`include/ck_tile/ops/gemm/block/block_universal_gemm_as_bs_cr.hpp`**
```
BLdsTile b_warp_tile_;
```

**`include/ck_tile/ops/gemm/warp/warp_gemm.hpp`**
```
using WarpGemmMfma_f32_16x16x128_fp8_fp8 = WarpGemmImpl<WarpGemmAtrributeMfma<
WarpGemmAttributeMfmaImpl_f32_16x16x128_fp8_fp8<WGAttrCtlEnum::Default_>>>;
using WarpGemmMfma_f32_16x16x128_fp8_bf8 = WarpGemmImpl<WarpGemmAtrributeMfma<
WarpGemmAttributeMfmaImpl_f32_16x16x128_fp8_bf8<WGAttrCtlEnum::Default_>>>;
```

**`include/ck_tile/ops/gemm/warp/warp_gemm_attribute_mfma_impl.hpp`**
```
template <typename AType_, typename BType_, WGAttrCtlEnum Ctrl_ = WGAttrCtlEnum::Default_>
struct WarpGemmAttributeMfmaImpl_f32_16x16x128_f8_bf8_base
static constexpr WGAttrCtlEnum Ctrl = Ctrl_;
using ADataType                     = AType_;
```
