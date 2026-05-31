# Diff summary

- **files changed:** 36
- **lines:** +474 / -158
- **kernel-ish files:** 34

## Files (by churn)

- `include/ck/utility/amd_buffer_addressing.hpp`  (+92/-55)
- `include/ck/tensor_operation/gpu/block/blockwise_gemm_xdlops.hpp`  (+70/-57)
- `include/ck/tensor_operation/gpu/warp/xdlops_gemm.hpp`  (+86/-12)
- `include/ck/utility/amd_xdlops.hpp`  (+65/-0)
- `example/01_gemm/gemm_xdl_fp8_bf8.cpp`  (+49/-0)
- `include/ck/tensor_operation/gpu/element/unary_element_wise_operation.hpp`  (+32/-0)
- `include/ck/tensor_operation/gpu/grid/gridwise_gemm_xdl_cshuffle_v1.hpp`  (+13/-11)
- `library/src/tensor_operation_instance/gpu/grouped_gemm_fixed_nk/CMakeLists.txt`  (+16/-8)
- `example/01_gemm/CMakeLists.txt`  (+11/-4)
- `include/ck/tensor_operation/gpu/grid/gridwise_batched_gemm_multiple_d_gemm_multiple_d_xdl_cshuffle_v1.hpp`  (+6/-1)
- `library/include/ck/library/reference_tensor_operation/cpu/reference_gemm.hpp`  (+4/-3)
- `include/ck/tensor_operation/gpu/device/impl/device_gemm_xdl_cshuffle.hpp`  (+4/-2)
- `library/include/ck/library/utility/host_tensor_generator.hpp`  (+2/-2)
- `include/ck/tensor_operation/gpu/grid/gridwise_batched_gemm_gemm_xdl_cshuffle_v1.hpp`  (+2/-1)
- `include/ck/tensor_operation/gpu/grid/gridwise_batched_gemm_multiple_d_softmax_gemm_xdl_cshuffle_v1.hpp`  (+2/-1)

## Key added lines (kernel files)

**`example/01_gemm/gemm_xdl_fp8_bf8.cpp`**
```
using ADataType        = ck::f8_t;
using BDataType        = ck::bf8_t;
using CDataType        = ck::f8_t;
using AccDataType      = float;
```

**`include/ck/tensor_operation/gpu/block/blockwise_gemm_xdlops.hpp`**
```
typename FloatA,
typename FloatB,
static constexpr auto xdlops_gemm = XdlopsGemm<FloatA, MPerXDL, NPerXDL, KPack, FloatB>{};
auto a_thread_buf = make_static_buffer<AddressSpaceEnum::Vgpr, FloatA>(
```

**`include/ck/tensor_operation/gpu/device/impl/device_gemm_xdl_cshuffle.hpp`**
```
typename ComputeTypeA       = CDataType,
typename ComputeTypeB       = ComputeTypeA>
ComputeTypeA,
ComputeTypeB>;
```

**`include/ck/tensor_operation/gpu/element/unary_element_wise_operation.hpp`**
```
template <>
__host__ __device__ void operator()<bf8_t, bf8_t>(bf8_t& y, const bf8_t& x) const
template <>
__host__ __device__ void operator()<float, bf8_t>(float& y, const bf8_t& x) const
```

**`include/ck/tensor_operation/gpu/grid/gemm_layernorm/gridwise_gemm_multiple_d_welford_first_half_xdl_cshuffle.hpp`**
```
ABDataType,
```
