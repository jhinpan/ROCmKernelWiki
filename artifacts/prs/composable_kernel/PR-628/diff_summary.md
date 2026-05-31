# Diff summary

- **files changed:** 13
- **lines:** +27 / -26
- **kernel-ish files:** 7

## Files (by churn)

- `include/ck/ck.hpp`  (+2/-10)
- `include/ck/tensor_operation/gpu/grid/gridwise_gemm_multiple_d_wmma_cshuffle.hpp`  (+7/-4)
- `Jenkinsfile`  (+2/-2)
- `CHANGELOG.md`  (+2/-1)
- `include/ck/tensor_operation/gpu/device/impl/device_batched_contraction_multiple_d_wmma_cshuffle.hpp`  (+2/-1)
- `include/ck/tensor_operation/gpu/device/impl/device_gemm_multiple_d_wmma_cshuffle.hpp`  (+2/-1)
- `include/ck/tensor_operation/gpu/device/impl/device_gemm_wmma.hpp`  (+2/-1)
- `include/ck/tensor_operation/gpu/device/impl/device_grouped_conv_fwd_multiple_d_wmma_cshuffle.hpp`  (+2/-1)
- `include/ck/tensor_operation/gpu/grid/gridwise_gemm_wmma.hpp`  (+2/-1)
- `example/01_gemm/CMakeLists.txt`  (+1/-1)
- `example/02_gemm_bilinear/CMakeLists.txt`  (+1/-1)
- `example/29_batched_gemm_bias_e_permute/CMakeLists.txt`  (+1/-1)
- `example/30_grouped_conv_fwd_multiple_d/CMakeLists.txt`  (+1/-1)

## Key added lines (kernel files)

**`include/ck/tensor_operation/gpu/device/impl/device_batched_contraction_multiple_d_wmma_cshuffle.hpp`**
```
if(ck::get_device_name() == "gfx1100" || ck::get_device_name() == "gfx1101" ||
ck::get_device_name() == "gfx1102")
```

**`include/ck/tensor_operation/gpu/device/impl/device_gemm_multiple_d_wmma_cshuffle.hpp`**
```
if(ck::get_device_name() == "gfx1100" || ck::get_device_name() == "gfx1101" ||
ck::get_device_name() == "gfx1102")
```

**`include/ck/tensor_operation/gpu/device/impl/device_gemm_wmma.hpp`**
```
if(ck::get_device_name() == "gfx1100" || ck::get_device_name() == "gfx1101" ||
ck::get_device_name() == "gfx1102")
```

**`include/ck/tensor_operation/gpu/device/impl/device_grouped_conv_fwd_multiple_d_wmma_cshuffle.hpp`**
```
if(get_device_name() == "gfx1100" || get_device_name() == "gfx1101" ||
ck::get_device_name() == "gfx1102")
```

**`include/ck/tensor_operation/gpu/grid/gridwise_gemm_multiple_d_wmma_cshuffle.hpp`**
```
defined(__gfx1102__))
defined(__gfx1102__))
defined(__gfx1102__))
```
