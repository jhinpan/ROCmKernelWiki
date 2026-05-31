# Diff summary

- **files changed:** 38
- **lines:** +57 / -259
- **kernel-ish files:** 26

## Files (by churn)

- `profiler/README.md`  (+0/-83)
- `Jenkinsfile`  (+17/-22)
- `example/02_gemm_bilinear/README.md`  (+0/-17)
- `example/15_grouped_gemm/README.md`  (+0/-16)
- `example/46_gemm_add_multiply/README.md`  (+0/-16)
- `example/01_gemm/README.md`  (+0/-14)
- `example/09_convnd_fwd/README.md`  (+0/-14)
- `example/04_gemm_add_add_fastgelu/README.md`  (+0/-13)
- `example/30_grouped_conv_fwd_multiple_d/README.md`  (+0/-12)
- `example/26_contraction/README.md`  (+0/-11)
- `test/grouped_convnd_bwd_weight/test_grouped_convnd_bwd_weight.cpp`  (+4/-4)
- `CMakeLists.txt`  (+3/-3)
- `include/ck/host_utility/device_prop.hpp`  (+3/-3)
- `include/ck/tensor_operation/gpu/device/impl/device_batched_gemm_softmax_gemm_permute_wmma_cshuffle.hpp`  (+2/-2)
- `include/ck/tensor_operation/gpu/device/impl/device_convnd_bwd_data_nwc_kxc_nwk_dl.hpp`  (+2/-2)

## Key added lines (kernel files)

**`include/ck/host_utility/device_prop.hpp`**
```
inline bool is_gfx101_supported()
inline bool is_gfx103_supported()
inline bool is_gfx11_supported()
```

**`include/ck/tensor_operation/gpu/device/impl/device_batched_contraction_multiple_d_wmma_cshuffle.hpp`**
```
if(ck::is_gfx11_supported())
```

**`include/ck/tensor_operation/gpu/device/impl/device_batched_gemm_multiple_d_dl.hpp`**
```
ck::is_gfx103_supported() || ck::is_gfx11_supported())
```

**`include/ck/tensor_operation/gpu/device/impl/device_batched_gemm_softmax_gemm_permute_wmma_cshuffle.hpp`**
```
if(ck::is_gfx11_supported())
if(ck::is_gfx11_supported())
```

**`include/ck/tensor_operation/gpu/device/impl/device_convnd_bwd_data_nwc_kxc_nwk_dl.hpp`**
```
if(!(ck::get_device_name() == "gfx906" || ck::is_gfx103_supported() ||
ck::is_gfx11_supported()))
```
