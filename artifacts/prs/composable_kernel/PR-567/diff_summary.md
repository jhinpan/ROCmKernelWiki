# Diff summary

- **files changed:** 15
- **lines:** +4852 / -7
- **kernel-ish files:** 11

## Files (by churn)

- `include/ck/tensor_operation/gpu/device/impl/device_batched_contraction_multiple_d_wmma_cshuffle.hpp`  (+991/-0)
- `include/ck/tensor_operation/gpu/grid/gridwise_gemm_multiple_d_wmma_cshuffle.hpp`  (+937/-0)
- `include/ck/tensor_operation/gpu/device/impl/device_grouped_conv_fwd_multiple_d_wmma_cshuffle.hpp`  (+850/-0)
- `include/ck/tensor_operation/gpu/device/impl/device_gemm_multiple_d_wmma_cshuffle.hpp`  (+654/-0)
- `example/29_batched_gemm_bias_e_permute/batched_gemm_bias_e_permute_wmma_fp16.cpp`  (+431/-0)
- `example/30_grouped_conv_fwd_multiple_d/common_wmma.hpp`  (+355/-0)
- `example/02_gemm_bilinear/gemm_bilinear_wmma_fp16.cpp`  (+304/-0)
- `example/30_grouped_conv_fwd_multiple_d/run_grouped_conv_fwd_bias_relu_add_wmma_example.inc`  (+286/-0)
- `example/30_grouped_conv_fwd_multiple_d/grouped_conv_fwd_bias_relu_add_wmma_fp16.cpp`  (+26/-0)
- `example/01_gemm/CMakeLists.txt`  (+5/-3)
- `example/46_gemm_add_multiply/run_gemm_add_multiply_example.inc`  (+2/-3)
- `example/29_batched_gemm_bias_e_permute/CMakeLists.txt`  (+4/-0)
- `example/02_gemm_bilinear/CMakeLists.txt`  (+3/-0)
- `example/30_grouped_conv_fwd_multiple_d/CMakeLists.txt`  (+3/-0)
- `example/30_grouped_conv_fwd_multiple_d/common.hpp`  (+1/-1)

## Key added lines (kernel files)

**`example/02_gemm_bilinear/gemm_bilinear_wmma_fp16.cpp`**
```
struct AlphaBetaAdd
AlphaBetaAdd(float alpha, float beta) : alpha_(alpha), beta_(beta){};
template <typename E, typename C, typename D>
__host__ __device__ constexpr void operator()(E& e, const C& c, const D& d) const;
```

**`example/29_batched_gemm_bias_e_permute/batched_gemm_bias_e_permute_wmma_fp16.cpp`**
```
template <ck::index_t... Is>
using S = ck::Sequence<Is...>;
using F16 = ck::half_t;
using F32 = float;
```

**`example/30_grouped_conv_fwd_multiple_d/common.hpp`**
```
num_dim_spatial, threshold_to_catch_partial_args + 1, argv);
```

**`example/30_grouped_conv_fwd_multiple_d/common_wmma.hpp`**
```
using BF16 = ck::bhalf_t;
using FP16 = ck::half_t;
using FP32 = float;
using I4 = ck::int4_t;
```

**`example/30_grouped_conv_fwd_multiple_d/grouped_conv_fwd_bias_relu_add_wmma_fp16.cpp`**
```
using InKernelDataType       = FP16;
using WeiKernelDataType      = FP16;
using AccDataType            = FP32;
using CShuffleDataType       = FP16;
```
