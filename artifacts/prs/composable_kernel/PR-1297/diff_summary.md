# Diff summary

- **files changed:** 10
- **lines:** +56 / -56
- **kernel-ish files:** 10

## Files (by churn)

- `include/ck_tile/core/config.hpp`  (+21/-9)
- `include/ck_tile/ops/gemm/warp/warp_gemm_attribute_mfma_impl.hpp`  (+10/-14)
- `include/ck_tile/core/numeric/float8.hpp`  (+10/-10)
- `include/ck/tensor_operation/gpu/grid/gridwise_gemm_xdl_cshuffle_v3.hpp`  (+4/-6)
- `include/ck/tensor_operation/gpu/grid/gridwise_gemm_xdl_cshuffle_v3_multi_abd.hpp`  (+4/-6)
- `include/ck/tensor_operation/gpu/grid/gridwise_fpAintB_gemm_wmma.hpp`  (+2/-3)
- `include/ck/tensor_operation/gpu/grid/gridwise_gemm_xdlops_splitk_lds_direct_load.hpp`  (+2/-3)
- `include/ck/tensor_operation/gpu/device/impl/device_contraction_multiple_d_xdl_cshuffle.hpp`  (+1/-2)
- `include/ck/tensor_operation/gpu/device/impl/device_grouped_gemm_multi_abd_xdl_fixed_nk.hpp`  (+1/-2)
- `include/ck_tile/core/tensor/tile_elementwise.hpp`  (+1/-1)

## Key added lines (kernel files)

**`include/ck_tile/core/config.hpp`**
```
defined(__gfx942__)
defined(__gfx1034__) || defined(__gfx1035__) || defined(__gfx1036__)
defined(__gfx9__) // for GPU code
```
