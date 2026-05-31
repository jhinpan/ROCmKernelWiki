# Diff summary

- **files changed:** 8
- **lines:** +365 / -200
- **kernel-ish files:** 8

## Files (by churn)

- `host/driver_offline/include/driver_gemm_xdlops_v2r3.hpp`  (+107/-33)
- `composable_kernel/include/tensor_operation/gridwise_gemm_xdlops_v2r4.hpp`  (+75/-61)
- `host/driver_offline/include/driver_gemm_xdlops_v2r4.hpp`  (+102/-33)
- `composable_kernel/include/tensor_operation/gridwise_gemm_xdlops_v2r3.hpp`  (+68/-54)
- `host/driver_offline/include/device_convolution_backward_weight_implicit_gemm_v4r4r5_xdlops_atomic_nhwc_kyxc_nhwk.hpp`  (+5/-7)
- `host/driver_offline/include/device_convolution_backward_weight_implicit_gemm_v4r4r2_xdlops_atomic_nchw_kcyx_nkhw.hpp`  (+3/-5)
- `host/driver_offline/include/device_convolution_backward_weight_implicit_gemm_v4r4r4_xdlops_atomic_nhwc_kyxc_nhwk.hpp`  (+3/-5)
- `host/driver_offline/include/device_convolution_forward_implicit_gemm_v4r4r4_xdlops_nhwc_kyxc_nhwk.hpp`  (+2/-2)

## Key added lines (kernel files)

**`composable_kernel/include/tensor_operation/gridwise_gemm_xdlops_v2r3.hpp`**
```
typename CBlockClusterAdaptor,
bool HasMainKBlockLoop>
GridwiseGemm::template Run<HasMainKBlockLoop>(p_a_grid,
p_b_grid,
```

**`composable_kernel/include/tensor_operation/gridwise_gemm_xdlops_v2r4.hpp`**
```
typename CBlockClusterAdaptor,
bool HasMainKBlockLoop>
GridwiseGemm::template Run<HasMainKBlockLoop>(p_a_grid,
p_b_grid,
```

**`host/driver_offline/include/device_convolution_backward_weight_implicit_gemm_v4r4r2_xdlops_atomic_nchw_kcyx_nkhw.hpp`**
```
const index_t GemmK0 =
math::integer_divide_ceil(GemmKTotal, GemmK1 * GemmKPerBlock * GemmKBatch) * GemmKPerBlock;
const index_t GemmKPad = GemmKBatch * GemmK0 * GemmK1;
```

**`host/driver_offline/include/device_convolution_backward_weight_implicit_gemm_v4r4r4_xdlops_atomic_nhwc_kyxc_nhwk.hpp`**
```
const index_t GemmK0 =
math::integer_divide_ceil(GemmKTotal, GemmK1 * GemmKPerBlock * GemmKBatch) * GemmKPerBlock;
const index_t GemmKPad = GemmKBatch * GemmK0 * GemmK1;
```

**`host/driver_offline/include/device_convolution_backward_weight_implicit_gemm_v4r4r5_xdlops_atomic_nhwc_kyxc_nhwk.hpp`**
```
const index_t GemmK0 =
math::integer_divide_ceil(GemmKTotal, GemmK1 * GemmKPerBlock * GemmKBatch) * GemmKPerBlock;
const index_t GemmKPad = GemmKBatch * GemmK0 * GemmK1;
```
