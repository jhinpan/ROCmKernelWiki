# Diff summary

- **files changed:** 82
- **lines:** +114 / -137
- **kernel-ish files:** 82

## Files (by churn)

- `include/ck/tensor_operation/gpu/device/impl/device_batched_gemm_softmax_gemm_permute_wmma_cshuffle.hpp`  (+3/-3)
- `include/ck/tensor_operation/gpu/grid/gridwise_gemm_multiple_d_wmma_cshuffle.hpp`  (+3/-3)
- `include/ck/tensor_operation/gpu/grid/gridwise_gemm_xdl_cshuffle_v1.hpp`  (+2/-4)
- `include/ck/tensor_operation/gpu/grid/gridwise_gemm_xdl_cshuffle_v2.hpp`  (+2/-4)
- `include/ck/tensor_operation/gpu/grid/gridwise_gemm_xdlops_v2r3.hpp`  (+2/-4)
- `include/ck/tensor_operation/gpu/device/impl/device_batched_gemm_multiple_d_dl.hpp`  (+2/-3)
- `include/ck/tensor_operation/gpu/device/impl/device_grouped_conv_bwd_weight_dl.hpp`  (+2/-3)
- `include/ck/tensor_operation/gpu/device/impl/device_grouped_conv_fwd_dl_multiple_d_nhwc_kyxc_nhwk.hpp`  (+2/-3)
- `include/ck/tensor_operation/gpu/device/impl/device_grouped_gemm_multiple_d_dl.hpp`  (+2/-3)
- `include/ck/tensor_operation/gpu/grid/gridwise_tensor_rearrange.hpp`  (+2/-3)
- `include/ck/tensor_operation/gpu/device/impl/device_batched_gemm_multiple_d_xdl_cshuffle_v3.hpp`  (+2/-2)
- `include/ck/tensor_operation/gpu/device/impl/device_batched_gemm_xdl_fpAintB_b_scale.hpp`  (+2/-2)
- `include/ck/tensor_operation/gpu/device/impl/device_gemm_multiple_d_dl.hpp`  (+2/-2)
- `include/ck/tensor_operation/gpu/device/impl/device_grouped_conv_bwd_weight_two_stage_xdl_cshuffle.hpp`  (+2/-2)
- `include/ck/tensor_operation/gpu/device/impl/device_grouped_conv_bwd_weight_xdl_cshuffle_v3.hpp`  (+2/-2)

## Key added lines (kernel files)

**`include/ck/tensor_operation/gpu/device/impl/device_batched_gemm_multiple_d_dl.hpp`**
```
defined(__gfx103__) || defined(__gfx11__) || defined(__gfx12__))
```

**`include/ck/tensor_operation/gpu/device/impl/device_gemm_multiple_d_dl.hpp`**
```
defined(__gfx12__))
```

**`include/ck/tensor_operation/gpu/device/impl/device_grouped_conv_bwd_data_multiple_d_xdl_cshuffle_v1.hpp`**
```
ignore = KBatch;
```

**`include/ck/tensor_operation/gpu/device/impl/device_grouped_conv_bwd_weight_dl.hpp`**
```
defined(__gfx94__) || defined(__gfx11__) || defined(__gfx12__))
```

**`include/ck/tensor_operation/gpu/device/impl/device_grouped_conv_fwd_dl_multiple_d_nhwc_kyxc_nhwk.hpp`**
```
defined(__gfx94__) || defined(__gfx11__) || defined(__gfx12__))
```
