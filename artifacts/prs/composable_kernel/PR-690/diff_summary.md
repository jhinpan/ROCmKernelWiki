# Diff summary

- **files changed:** 47
- **lines:** +228 / -72
- **kernel-ish files:** 43

## Files (by churn)

- `include/ck/tensor_operation/gpu/warp/xdlops_gemm.hpp`  (+59/-1)
- `include/ck/utility/amd_xdlops.hpp`  (+39/-1)
- `include/ck/ck.hpp`  (+11/-7)
- `include/ck/tensor_operation/gpu/device/device_gemm_xdl_waveletmodel_cshuffle.hpp`  (+4/-2)
- `include/ck/tensor_operation/gpu/device/device_grouped_gemm_softmax_gemm_permute_xdl_cshuffle.hpp`  (+4/-2)
- `include/ck/tensor_operation/gpu/device/device_splitk_contraction_multiple_d_xdl_cshuffle.hpp`  (+4/-2)
- `include/ck/tensor_operation/gpu/device/impl/device_batched_contraction_multiple_d_xdl_cshuffle.hpp`  (+4/-2)
- `include/ck/tensor_operation/gpu/device/impl/device_batched_gemm_gemm_xdl_cshuffle.hpp`  (+4/-2)
- `include/ck/tensor_operation/gpu/device/impl/device_batched_gemm_multi_d_xdl.hpp`  (+4/-2)
- `include/ck/tensor_operation/gpu/device/impl/device_batched_gemm_multiple_d_gemm_multiple_d_xdl_cshuffle.hpp`  (+4/-2)
- `include/ck/tensor_operation/gpu/device/impl/device_batched_gemm_softmax_gemm_permute_xdl_cshuffle.hpp`  (+4/-2)
- `include/ck/tensor_operation/gpu/device/impl/device_batched_gemm_softmax_gemm_xdl_cshuffle.hpp`  (+4/-2)
- `include/ck/tensor_operation/gpu/device/impl/device_contraction_multiple_d_xdl_cshuffle.hpp`  (+4/-2)
- `include/ck/tensor_operation/gpu/device/impl/device_gemm_bias_e_permute_xdl.hpp`  (+4/-2)
- `include/ck/tensor_operation/gpu/device/impl/device_gemm_multiple_d_layernorm_xdl_cshuffle.hpp`  (+4/-2)

## Key added lines (kernel files)

**`include/ck/ck.hpp`**
```
defined(__gfx90a__) || defined(__gfx940__) // for GPU code
defined(__gfx940__) // for GPU code
```

**`include/ck/tensor_operation/gpu/device/device_gemm_xdl_waveletmodel_cshuffle.hpp`**
```
defined(__gfx940__))
if(!(ck::get_device_name() == "gfx908" || ck::get_device_name() == "gfx90a" ||
ck::get_device_name() == "gfx940"))
```

**`include/ck/tensor_operation/gpu/device/device_grouped_gemm_softmax_gemm_permute_xdl_cshuffle.hpp`**
```
defined(__gfx940__))
if(!(ck::get_device_name() == "gfx908" || ck::get_device_name() == "gfx90a" ||
ck::get_device_name() == "gfx940"))
```

**`include/ck/tensor_operation/gpu/device/device_splitk_contraction_multiple_d_xdl_cshuffle.hpp`**
```
defined(__gfx940__))
if(!(ck::get_device_name() == "gfx908" || ck::get_device_name() == "gfx90a" ||
ck::get_device_name() == "gfx940"))
```

**`include/ck/tensor_operation/gpu/device/impl/device_batched_contraction_multiple_d_xdl_cshuffle.hpp`**
```
defined(__gfx940__))
if(!(ck::get_device_name() == "gfx908" || ck::get_device_name() == "gfx90a" ||
ck::get_device_name() == "gfx940"))
```
