# Diff summary

- **files changed:** 123
- **lines:** +849 / -575
- **kernel-ish files:** 123

## Files (by churn)

- `test/grouped_convnd_fwd/test_grouped_convnd_fwd_multi_ab_interface.cpp`  (+17/-14)
- `test/grouped_gemm/test_grouped_gemm_interface_xdl.cpp`  (+16/-13)
- `test/grouped_gemm/test_grouped_gemm_util.hpp`  (+17/-10)
- `example/41_grouped_conv_conv_fwd/grouped_conv_conv_fwd_xdl_bf16.cpp`  (+16/-8)
- `test/wrapper/test_wrapper_gemm_xdl.cpp`  (+23/-1)
- `example/03_gemm_bias_relu/gemm_bias_relu_xdl_fp16.cpp`  (+11/-11)
- `example/09_convnd_fwd/convnd_fwd_xdl_bf8.cpp`  (+15/-7)
- `example/09_convnd_fwd/convnd_fwd_xdl_bf8_fp8.cpp`  (+15/-7)
- `example/09_convnd_fwd/convnd_fwd_xdl_fp16_comp_fp8.cpp`  (+15/-7)
- `example/09_convnd_fwd/convnd_fwd_xdl_fp8.cpp`  (+15/-7)
- `example/09_convnd_fwd/convnd_fwd_xdl_fp8_bf8.cpp`  (+15/-7)
- `example/62_convnd_activ/binary/convnd_bwd_weight_xdl_bilinear_residual_fp16.cpp`  (+15/-7)
- `example/47_gemm_bias_softmax_gemm_permute/gemm_bias_softmax_gemm_permute_xdl.cpp`  (+13/-7)
- `example/65_gemm_multiply_multiply/gemm_multiply_multiply_xdl_fp16_bpreshuffle.cpp`  (+13/-6)
- `example/01_gemm/gemm_xdl_fp16_pk_i4_v3_b_scale.cpp`  (+9/-8)

## Key added lines (kernel files)

**`example/01_gemm/gemm_xdl_bf16_pk_i4_v3.cpp`**
```
if(!(ck::get_device_name() == "gfx942" || ck::get_device_name() == "gfx950" ||
ck::is_gfx11_supported() || ck::is_gfx12_supported()))
std::cout << "This kernel support gfx942, gfx950, gfx11 and gfx12 only" << std::endl;
```

**`example/01_gemm/gemm_xdl_fp16_pk_i4_v3.cpp`**
```
if(!(ck::get_device_name() == "gfx942" || ck::get_device_name() == "gfx950" ||
ck::is_gfx11_supported() || ck::is_gfx12_supported()))
std::cout << "This kernel support gfx942, gfx950, gfx11 and gfx12 only" << std::endl;
```

**`example/01_gemm/gemm_xdl_fp16_pk_i4_v3_b_scale.cpp`**
```
KPerBlock, 8, 16,
16,   16,
2, 16, 16, 0,
1, 1, S<1, 16, 1, 16>, 4,
```

**`example/01_gemm/gemm_xdl_fp8_pk_i4_bpreshuffle_v3.cpp`**
```
static constexpr int KPack     = 32; // int4 -> 32, fp8 -> 16, fp16 -> 8
128, 16, KPack,
16,   16,
1, 1, S<1, 32, 1, 8>, 4,
```

**`example/01_gemm/gemm_xdl_fp8_pk_i4_v3.cpp`**
```
KPerBlock, 16, 16,
16,   16,
2, 16, 16, 0,
1, 1, S<1, 32, 1, 8>, 4,
```
