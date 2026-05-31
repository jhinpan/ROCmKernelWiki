# Diff summary

- **files changed:** 19
- **lines:** +1006 / -607
- **kernel-ish files:** 16

## Files (by churn)

- `include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_v1_mx.hpp`  (+270/-274)
- `include/ck/tensor_operation/gpu/block/blockwise_gemm_mx_pipeline_xdlops_base.hpp`  (+363/-0)
- `include/ck/tensor_operation/gpu/grid/gridwise_gemm_xdl_cshuffle_v3_mx.hpp`  (+65/-61)
- `example/67_gemm_microscaling/gemm_mx_fp8.cpp`  (+98/-0)
- `test/mx_mfma_op/mx_mfma_op.hpp`  (+60/-34)
- `include/ck/tensor_operation/gpu/warp/xdlops_gemm.hpp`  (+74/-15)
- `example/67_gemm_microscaling/gemm_mx_common.hpp`  (+14/-65)
- `example/67_gemm_microscaling/gemm_mx_fp8_e8m0_scale.cpp`  (+0/-42)
- `example/67_gemm_microscaling/gemm_mx_fp8_fp16_scale.cpp`  (+0/-42)
- `example/67_gemm_microscaling/gemm_mx_fp8_fp8_scale.cpp`  (+0/-42)
- `include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_mx_selector.hpp`  (+32/-3)
- `include/ck/utility/amd_xdlops.hpp`  (+12/-4)
- `include/ck/tensor_operation/gpu/device/impl/device_gemm_xdl_cshuffle_v3_mx.hpp`  (+6/-8)
- `example/67_gemm_microscaling/CMakeLists.txt`  (+2/-7)
- `example/67_gemm_microscaling/README.md`  (+4/-4)

## Key added lines (kernel files)

**`example/67_gemm_microscaling/gemm_mx_common.hpp`**
```
<< "arg5 to 10: M(128x), N(128x), K(256x), StrideA, StrideB, StrideC" << std::endl
template <typename DeviceOpInstance,
typename ADataType,
ck::index_t ScaleBlockSize>
```

**`example/67_gemm_microscaling/gemm_mx_fp8.cpp`**
```
using ADataType = ck::f8_t;
using BDataType = ck::f8_t;
using XDataType = ck::e8m0_bexp_t;
using CDataType        = ck::half_t;
```

**`include/ck/tensor_operation/gpu/block/blockwise_gemm_mx_pipeline_xdlops_base.hpp`**
```
namespace ck {
template <index_t BlockSize,
typename ADataType,
typename BDataType,
```

**`include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_mx_selector.hpp`**
```
template <typename T>
static constexpr bool is_scale_mfma_data_type()
return is_same_v<T, f8_ocp_t> || is_same_v<T, bf8_ocp_t> || is_same_v<T, f6_t> ||
is_same_v<T, bf6_t> || is_same_v<T, f4_t>;
```

**`include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_v1_mx.hpp`**
```
: BlockwiseGemmXdlops_mx_pipeline_base<ThreadBlockSize,
ADataType,
BDataType,
ATileDesc,
```
