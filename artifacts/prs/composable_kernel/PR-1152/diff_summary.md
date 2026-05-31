# Diff summary

- **files changed:** 69
- **lines:** +193 / -193
- **kernel-ish files:** 69

## Files (by churn)

- `include/ck/ck.hpp`  (+27/-16)
- `include/ck/tensor_operation/gpu/device/impl/device_grouped_gemm_multiple_d_dl.hpp`  (+16/-19)
- `include/ck/utility/type_convert.hpp`  (+17/-13)
- `include/ck/utility/amd_wmma.hpp`  (+13/-10)
- `include/ck/utility/amd_xdlops.hpp`  (+13/-9)
- `include/ck/host_utility/device_prop.hpp`  (+19/-0)
- `include/ck/tensor_operation/gpu/device/impl/device_batched_gemm_multiple_d_dl.hpp`  (+4/-8)
- `include/ck/tensor_operation/gpu/device/impl/device_gemm_multiple_d_dl.hpp`  (+4/-8)
- `include/ck/tensor_operation/gpu/device/impl/device_grouped_conv_fwd_dl_multiple_d_nhwc_kyxc_nhwk.hpp`  (+4/-8)
- `include/ck/tensor_operation/gpu/grid/gridwise_gemm_multiple_d_wmma_cshuffle.hpp`  (+4/-7)
- `include/ck/tensor_operation/gpu/device/impl/device_grouped_conv_fwd_dl_nhwc_kyxc_nhwk.hpp`  (+4/-5)
- `include/ck/tensor_operation/gpu/device/impl/device_convnd_bwd_data_nwc_kxc_nwk_dl.hpp`  (+2/-3)
- `include/ck/tensor_operation/gpu/device/impl/device_gemm_dl.hpp`  (+2/-3)
- `include/ck/tensor_operation/gpu/device/impl/device_grouped_conv_bwd_weight_dl.hpp`  (+2/-3)
- `include/ck/tensor_operation/gpu/device/impl/device_grouped_conv_fwd_multiple_abd_xdl_cshuffle.hpp`  (+2/-3)

## Key added lines (kernel files)

**`include/ck/ck.hpp`**
```
defined(__gfx1034__) || defined(__gfx1035__) || defined(__gfx1036__)
defined(__gfx90a__) || defined(__gfx94__)
defined(__gfx94__) // for GPU code
```

**`include/ck/host_utility/device_prop.hpp`**
```
inline bool is_navi1_supported()
return ck::get_device_name() == "gfx1010" || ck::get_device_name() == "gfx1011" ||
ck::get_device_name() == "gfx1012";
inline bool is_navi2_supported()
```

**`include/ck/tensor_operation/gpu/device/impl/device_batched_contraction_multiple_d_wmma_cshuffle.hpp`**
```
if(ck::is_navi3_supported())
```

**`include/ck/tensor_operation/gpu/device/impl/device_batched_contraction_multiple_d_xdl_cshuffle.hpp`**
```
defined(__gfx94__))
```

**`include/ck/tensor_operation/gpu/device/impl/device_batched_gemm_e_permute_xdl.hpp`**
```
defined(__gfx94__))
```
