# Diff summary

- **files changed:** 19
- **lines:** +4119 / -0
- **kernel-ish files:** 14

## Files (by churn)

- `include/ck/tensor_operation/gpu/grid/gridwise_batched_gemm_multiple_d_gemm_multiple_d_xdl_cshuffle_v1.hpp`  (+1268/-0)
- `include/ck/tensor_operation/gpu/device/device_batched_gemm_multiple_d_gemm_multiple_d_xdl_cshuffle.hpp`  (+951/-0)
- `example/37_batched_gemm_add_add_relu_gemm_add/batched_gemm_add_add_relu_gemm_add_xdl_fp16.cpp`  (+519/-0)
- `profiler/include/profile_batched_gemm_add_relu_gemm_add_impl.hpp`  (+360/-0)
- `profiler/src/profile_batched_gemm_add_relu_gemm_add.cpp`  (+209/-0)
- `profiler/src/profile_batched_gemm_gemm.cpp`  (+181/-0)
- `include/ck/tensor_operation/gpu/device/matrix_padder.hpp`  (+159/-0)
- `library/include/ck/library/tensor_operation_instance/gpu/batched_gemm_add_relu_gemm_add.hpp`  (+139/-0)
- `library/src/tensor_operation_instance/gpu/batched_gemm_add_relu_gemm_add/device_batched_gemm_add_relu_gemm_add_xdl_cshuffle_f16_f16_f16_f16_gmk_gnk_gon_gmo_instance.cpp`  (+81/-0)
- `library/src/tensor_operation_instance/gpu/batched_gemm_add_relu_gemm_add/device_batched_gemm_add_relu_gemm_add_xdl_cshuffle_f16_f16_f16_f16_gmk_gnk_gno_gmo_instance.cpp`  (+80/-0)
- `include/ck/tensor_operation/gpu/device/device_batched_gemm_multiple_d_gemm_multiple_d.hpp`  (+72/-0)
- `include/ck/tensor_operation/gpu/element/binary_element_wise_operation.hpp`  (+55/-0)
- `include/ck/tensor_operation/gpu/element/unary_element_wise_operation.hpp`  (+21/-0)
- `profiler/src/profiler.cpp`  (+12/-0)
- `library/src/tensor_operation_instance/gpu/batched_gemm_add_relu_gemm_add/CMakeLists.txt`  (+4/-0)

## Key added lines (kernel files)

**`example/37_batched_gemm_add_add_relu_gemm_add/batched_gemm_add_add_relu_gemm_add_xdl_fp16.cpp`**
```
Computes C_m_o = Relu(A0[m, k] * B0[n, k] + D00[m, n] + D01[mn]) * B1[n, o] + D1[m, o]
template <ck::index_t... Is>
using S = ck::Sequence<Is...>;
using F16 = ck::half_t;
```

**`include/ck/tensor_operation/gpu/device/device_batched_gemm_multiple_d_gemm_multiple_d.hpp`**
```
namespace ck {
namespace tensor_operation {
namespace device {
template <typename A0Layout,
```

**`include/ck/tensor_operation/gpu/device/device_batched_gemm_multiple_d_gemm_multiple_d_xdl_cshuffle.hpp`**
```
namespace ck {
namespace tensor_operation {
namespace device {
template <typename GridwiseGemm,
```

**`include/ck/tensor_operation/gpu/device/matrix_padder.hpp`**
```
template <bool PadM,
bool PadN,
bool PadK,
typename MPerTileType,
```

**`include/ck/tensor_operation/gpu/element/binary_element_wise_operation.hpp`**
```
template <>
__host__ __device__ constexpr void
operator()<float>(float& y, const float& x0, const half_t& x1) const
y = x0 + type_convert<half_t>(x1);
```
