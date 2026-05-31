# Diff summary

- **files changed:** 19
- **lines:** +1705 / -383
- **kernel-ish files:** 15

## Files (by churn)

- `include/ck/tensor_operation/gpu/device/impl/device_gemm_multiple_d_wmma_cshuffle_v3_blockscale_bpreshuffle.hpp`  (+360/-0)
- `example/65_gemm_multiply_multiply/gemm_multiply_multiply_wmma_fp8_blockscale_bpreshuffle.cpp`  (+357/-0)
- `include/ck/tensor_operation/gpu/device/impl/device_gemm_multiple_d_wmma_cshuffle_v3_b_preshuffle.hpp`  (+3/-305)
- `include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_wmmaops_v1.hpp`  (+244/-1)
- `include/ck/tensor_operation/gpu/device/impl/device_gemm_wmma_cshuffle_v3_common.hpp`  (+155/-33)
- `include/ck/tensor_operation/gpu/device/device_gemm_multiple_d_ab_scale.hpp`  (+169/-0)
- `library/include/ck/library/tensor_operation_instance/gpu/gemm_blockscale_wp.hpp`  (+147/-0)
- `include/ck/tensor_operation/gpu/grid/gridwise_gemm_wmma_cshuffle_v3_ab_scale.hpp`  (+52/-34)
- `library/src/tensor_operation_instance/gpu/gemm_blockscale_wp/device_gemm_blockscale_wp_wmma_f8_f8_bf16/device_gemm_blockscale_wp_wmma_f8_f8_bf16_mk_nk_mn_128_128_128.hpp`  (+77/-0)
- `include/ck/tensor_operation/gpu/grid/gridwise_gemm_wmma_cshuffle_v3_common.hpp`  (+43/-1)
- `library/src/tensor_operation_instance/gpu/gemm_blockscale_wp/device_gemm_blockscale_wp_wmma_f8_f8_bf16/device_gemm_blockscale_wp_wmma_f8_f8_bf16_mk_nk_mn_128_128_128_comp_default_instance.cpp`  (+38/-0)
- `library/src/tensor_operation_instance/gpu/gemm_blockscale_wp/device_gemm_blockscale_wp_wmma_f8_f8_bf16/device_gemm_blockscale_wp_wmma_f8_f8_bf16_mk_nk_mn_128_128_128_mem_default_instance.cpp`  (+38/-0)
- `include/ck/tensor_operation/gpu/grid/gridwise_ab_transfer_thread_tiles.hpp`  (+5/-5)
- `library/src/tensor_operation_instance/gpu/CMakeLists.txt`  (+9/-0)
- `library/src/tensor_operation_instance/gpu/gemm_blockscale_wp/CMakeLists.txt`  (+4/-1)

## Key added lines (kernel files)

**`example/65_gemm_multiply_multiply/gemm_multiply_multiply_wmma_fp8_blockscale_bpreshuffle.cpp`**
```
template <ck::index_t... Is>
using S = ck::Sequence<Is...>;
using BF16 = ck::bhalf_t;
using FP8  = ck::f8_t;
```

**`include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_wmmaops_v1.hpp`**
```
typename BScaleStruct,
typename enable_if<ck::is_same_v<AScaleStruct, Empty>, bool>::type = false>
template <bool HasMainLoop,
TailNumber TailNum,
```

**`include/ck/tensor_operation/gpu/device/device_gemm_multiple_d_ab_scale.hpp`**
```
template <typename ALayout,
typename BLayout,
typename DsLayout,
typename ELayout,
```

**`include/ck/tensor_operation/gpu/device/impl/device_gemm_multiple_d_wmma_cshuffle_v3_b_preshuffle.hpp`**
```
ComputeTypeB,
true>; // IsBPreshuffle
using Invoker = typename DeviceGemmCommon::Invoker;
```

**`include/ck/tensor_operation/gpu/device/impl/device_gemm_multiple_d_wmma_cshuffle_v3_blockscale_bpreshuffle.hpp`**
```
namespace ck {
namespace tensor_operation {
namespace device {
template <typename ALayout,
```
