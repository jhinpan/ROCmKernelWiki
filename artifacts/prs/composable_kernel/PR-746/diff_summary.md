# Diff summary

- **files changed:** 19
- **lines:** +1194 / -44
- **kernel-ish files:** 16

## Files (by churn)

- `example/02_gemm_bilinear/gemm_bilinear_wmma_int8.cpp`  (+304/-0)
- `include/ck/tensor_operation/gpu/block/blockwise_gemm_wmma.hpp`  (+94/-41)
- `library/include/ck/library/tensor_operation_instance/gpu/grouped_conv_fwd/device_grouped_conv2d_fwd_wmma_instance.hpp`  (+134/-0)
- `library/src/tensor_operation_instance/gpu/gemm_bilinear/device_gemm_bilinear_wmma_c_shuffle_i8_i8_i8_i8_mk_nk_mn_mn_instance.cpp`  (+115/-0)
- `library/src/tensor_operation_instance/gpu/gemm_bilinear/device_gemm_bilinear_wmma_c_shuffle_i8_i8_i8_i8_km_kn_mn_mn_instance.cpp`  (+89/-0)
- `library/src/tensor_operation_instance/gpu/gemm_bilinear/device_gemm_bilinear_wmma_c_shuffle_i8_i8_i8_i8_km_nk_mn_mn_instance.cpp`  (+89/-0)
- `library/src/tensor_operation_instance/gpu/gemm_bilinear/device_gemm_bilinear_wmma_c_shuffle_i8_i8_i8_i8_mk_kn_mn_mn_instance.cpp`  (+89/-0)
- `library/include/ck/library/tensor_operation_instance/gpu/gemm_bilinear.hpp`  (+76/-0)
- `library/src/tensor_operation_instance/gpu/grouped_conv2d_fwd/device_grouped_conv2d_fwd_wmma_gnhwc_gkyxc_gnhwk_f16_instance.cpp`  (+66/-0)
- `library/src/tensor_operation_instance/gpu/grouped_conv2d_fwd/device_grouped_conv2d_fwd_wmma_gnhwc_gkyxc_gnhwk_i8_instance.cpp`  (+66/-0)
- `library/include/ck/library/tensor_operation_instance/gpu/grouped_convolution_forward.hpp`  (+32/-0)
- `profiler/src/profile_gemm_bilinear.cpp`  (+19/-0)
- `include/ck/tensor_operation/gpu/element/binary_element_wise_operation.hpp`  (+7/-0)
- `include/ck/tensor_operation/gpu/device/impl/device_grouped_conv_fwd_multiple_d_wmma_cshuffle.hpp`  (+2/-2)
- `library/src/tensor_operation_instance/gpu/gemm_bilinear/CMakeLists.txt`  (+4/-0)

## Key added lines (kernel files)

**`example/02_gemm_bilinear/gemm_bilinear_wmma_int8.cpp`**
```
struct AlphaBetaAdd
AlphaBetaAdd(int alpha, int beta) : alpha_(alpha), beta_(beta){};
template <typename E, typename C, typename D>
__host__ __device__ constexpr void operator()(E& e, const C& c, const D& d) const;
```

**`include/ck/tensor_operation/gpu/block/blockwise_gemm_wmma.hpp`**
```
if constexpr(MRepeat < NRepeat)
static_for<0, KPerBlock / WmmaK, 1>{}(
[&](auto k) { // k=0,1,2 instead of k=0,kpack*1, ...
static_for<0, MRepeat, 1>{}([&](auto m0) {
```

**`include/ck/tensor_operation/gpu/device/impl/device_grouped_conv_fwd_multiple_d_wmma_cshuffle.hpp`**
```
const index_t X          = arg.b_g_k_c_xs_lengths_[i + 3];
const index_t X        = arg.b_g_k_c_xs_lengths_[i + 3];
```

**`include/ck/tensor_operation/gpu/element/binary_element_wise_operation.hpp`**
```
template <>
__host__ __device__ constexpr void operator()<std::int8_t, std::int32_t, std::int8_t>(
std::int8_t& y, const std::int32_t& x0, const std::int8_t& x1) const
y = type_convert<std::int8_t>(x0 + ck::type_convert<std::int32_t>(x1));
```

**`library/include/ck/library/tensor_operation_instance/device_operation_instance_factory.hpp`**
```
using I8_Tuple      = ck::Tuple<I8>;
```
