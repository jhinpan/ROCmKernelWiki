# Diff summary

- **files changed:** 13
- **lines:** +135 / -49
- **kernel-ish files:** 11

## Files (by churn)

- `include/ck_tile/host/tensor_shuffle_utils.hpp`  (+75/-23)
- `test/ck_tile/gemm_block_scale/test_gemm_quant_fixtures.hpp`  (+17/-7)
- `test/ck_tile/gemm_block_scale/test_gemm_quant_base.hpp`  (+12/-2)
- `include/ck_tile/ops/gemm_quant/block/block_universal_gemm_as_aquant_bs_cr.hpp`  (+5/-6)
- `example/ck_tile/38_block_scale_gemm/gemm_utils.hpp`  (+8/-0)
- `include/ck_tile/ops/gemm_quant/block/block_universal_gemm_as_bs_bquant_cr.hpp`  (+2/-4)
- `include/ck_tile/ops/gemm_quant/kernel/gemm_quant_kernel.hpp`  (+4/-1)
- `include/ck_tile/ops/gemm_quant/pipeline/tile_gemm_quant_traits.hpp`  (+3/-2)
- `example/ck_tile/38_block_scale_gemm/gemm_quant_basic.cpp`  (+4/-0)
- `include/ck_tile/ops/gemm_quant/block/block_universal_gemm_ar_flatbr_bquant_cr.hpp`  (+2/-2)
- `example/ck_tile/38_block_scale_gemm/CMakeLists.txt`  (+1/-1)
- `test/ck_tile/gemm_block_scale/CMakeLists.txt`  (+1/-1)
- `include/ck_tile/ops/gemm/warp/warp_gemm_attribute_wmma.hpp`  (+1/-0)

## Key added lines (kernel files)

**`example/ck_tile/38_block_scale_gemm/gemm_quant_basic.cpp`**
```
return !run_gemm_example<GemmConfigBQuantPrefill_Wmma>(argc, argv);
```

**`example/ck_tile/38_block_scale_gemm/gemm_utils.hpp`**
```
template <typename PrecType>
struct GemmConfigBQuantPrefill_Wmma : public GemmConfigBQuantPrefill<PrecType>
static constexpr ck_tile::index_t M_Warp_Tile = 16;
static constexpr ck_tile::index_t N_Warp_Tile = 16;
```

**`include/ck_tile/host/tensor_shuffle_utils.hpp`**
```
int n_ = t.get_lengths()[1];
int k_ = t.get_lengths()[0];
if(ck_tile::is_gfx12_supported())
constexpr int divisor      = 2;
```

**`include/ck_tile/ops/gemm/warp/warp_gemm_attribute_wmma.hpp`**
```
static constexpr index_t kCMLane     = Impl::kCMLane;
```

**`include/ck_tile/ops/gemm_quant/block/block_universal_gemm_ar_flatbr_bquant_cr.hpp`**
```
scale_reg_f = __builtin_amdgcn_cvt_f32_fp8(static_cast<uint32_t>(scale), 0);
scale_reg_f = __builtin_amdgcn_cvt_f32_bf8(static_cast<uint32_t>(scale), 0);
```
