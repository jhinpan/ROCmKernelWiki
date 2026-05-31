# Diff summary

- **files changed:** 33 (diff was byte-capped; summary is partial)
- **lines:** +5176 / -199
- **kernel-ish files:** 28

## Files (by churn)

- `include/ck/tensor_operation/gpu/device/impl/device_grouped_gemm_multi_abd_xdl_fixed_nk.hpp`  (+851/-0)
- `library/include/ck/library/tensor_operation_instance/gpu/grouped_gemm_multi_abd_fixed_nk.hpp`  (+470/-0)
- `library/include/ck/library/tensor_operation_instance/gpu/gemm_multi_abd.hpp`  (+468/-0)
- `example/59_grouped_gemm_multi_ABD/grouped_gemm_multi_abd_xdl_fixed_nk_bias_bf16_i8.cpp`  (+401/-0)
- `example/59_grouped_gemm_multi_ABD/grouped_gemm_multi_abd_xdl_fixed_nk_bias_fp16.cpp`  (+397/-0)
- `client_example/31_grouped_gemm_multi_abd/grouped_gemm_bias_fastgelu_xdl_bf16_i8.cpp`  (+286/-0)
- `client_example/31_grouped_gemm_multi_abd/grouped_gemm_fastgelu_xdl_bf16_i8.cpp`  (+282/-0)
- `example/60_gemm_multi_ABD/gemm_multi_ABD_xdl_bf16_i8.cpp`  (+270/-0)
- `client_example/30_gemm_multi_abd/gemm_bias_fastgelu_xdl_bf16_i8.cpp`  (+262/-0)
- `client_example/30_gemm_multi_abd/gemm_bias_xdl_bf16_i8.cpp`  (+262/-0)
- `client_example/30_gemm_multi_abd/gemm_xdl_gelu_bf16_i8.cpp`  (+261/-0)
- `client_example/30_gemm_multi_abd/gemm_xdl_bf16_i8.cpp`  (+257/-0)
- `include/ck/tensor_operation/gpu/thread/threadwise_tensor_slice_transfer_v7r2.hpp`  (+225/-22)
- `include/ck/tensor_operation/gpu/device/impl/device_gemm_multiple_abd_xdl_cshuffle.hpp`  (+15/-91)
- `include/ck/tensor_operation/gpu/device/device_grouped_gemm_multi_abd.hpp`  (+98/-0)

## Key added lines (kernel files)

**`client_example/30_gemm_multi_abd/gemm_bias_fastgelu_xdl_bf16_i8.cpp`**
```
template <ck::index_t... Is>
using S = ck::Sequence<Is...>;
using BF16 = ck::bhalf_t;
using I8   = int8_t;
```

**`client_example/30_gemm_multi_abd/gemm_bias_xdl_bf16_i8.cpp`**
```
template <ck::index_t... Is>
using S = ck::Sequence<Is...>;
using BF16 = ck::bhalf_t;
using I8   = int8_t;
```

**`client_example/30_gemm_multi_abd/gemm_xdl_bf16_i8.cpp`**
```
template <ck::index_t... Is>
using S = ck::Sequence<Is...>;
using BF16 = ck::bhalf_t;
using I8   = int8_t;
```

**`client_example/30_gemm_multi_abd/gemm_xdl_gelu_bf16_i8.cpp`**
```
template <ck::index_t... Is>
using S = ck::Sequence<Is...>;
using BF16 = ck::bhalf_t;
using I8   = int8_t;
```

**`client_example/31_grouped_gemm_multi_abd/grouped_gemm_bias_fastgelu_xdl_bf16_i8.cpp`**
```
template <ck::index_t... Is>
using S = ck::Sequence<Is...>;
using BF16 = ck::bhalf_t;
using I8   = int8_t;
```
