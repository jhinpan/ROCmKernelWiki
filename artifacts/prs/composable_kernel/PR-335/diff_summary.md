# Diff summary

- **files changed:** 14
- **lines:** +2950 / -912
- **kernel-ish files:** 11

## Files (by churn)

- `include/ck/tensor_operation/gpu/grid/gridwise_gemm_multiple_d_multiple_r_xdl_cshuffle.hpp`  (+901/-0)
- `include/ck/tensor_operation/gpu/device/device_gemm_multiple_d_multiple_r_xdl_cshuffle.hpp`  (+873/-0)
- `example/21_gemm_layernorm/gemm_bias_relu_add_layernorm_xdl_fp16.cpp`  (+160/-177)
- `example/16_gemm_reduce/gemm_reduce_xdl_mean_squaremean_fp16.cpp`  (+0/-314)
- `example/21_gemm_layernorm/gemm_layernorm_xdl_fp16.cpp`  (+138/-142)
- `example/16_gemm_multi_d_multi_reduces/gemm_add_add_mean_meansquare_xdl_fp16.cpp`  (+279/-0)
- `example/16_gemm_reduce/gemm_reduce_xdl_max_fp16.cpp`  (+0/-276)
- `example/16_gemm_multi_d_multi_reduces/gemm_mean_meansquare_xdl_fp16.cpp`  (+254/-0)
- `example/16_gemm_multi_d_multi_reduces/gemm_max_xdl_fp16.cpp`  (+227/-0)
- `include/ck/tensor_operation/gpu/device/device_gemm_multiple_d_multiple_r.hpp`  (+85/-0)
- `include/ck/tensor_operation/gpu/element/element_wise_operation.hpp`  (+29/-0)
- `example/16_gemm_multi_d_multi_reduces/CMakeLists.txt`  (+3/-0)
- `example/16_gemm_reduce/CMakeLists.txt`  (+0/-2)
- `example/CMakeLists.txt`  (+1/-1)

## Key added lines (kernel files)

**`example/16_gemm_multi_d_multi_reduces/gemm_add_add_mean_meansquare_xdl_fp16.cpp`**
```
template <ck::index_t... Is>
using S = ck::Sequence<Is...>;
using F16 = ck::half_t;
using F32 = float;
```

**`example/16_gemm_multi_d_multi_reduces/gemm_max_xdl_fp16.cpp`**
```
template <ck::index_t... Is>
using S = ck::Sequence<Is...>;
using F16 = ck::half_t;
using F32 = float;
```

**`example/16_gemm_multi_d_multi_reduces/gemm_mean_meansquare_xdl_fp16.cpp`**
```
template <ck::index_t... Is>
using S = ck::Sequence<Is...>;
using F16 = ck::half_t;
using F32 = float;
```

**`example/21_gemm_layernorm/gemm_bias_relu_add_layernorm_xdl_fp16.cpp`**
```
using CShuffleDataType         = F32;
using D0DataType               = F16;
using D1DataType               = F16;
using DsDataType               = ck::Tuple<D0DataType, D1DataType>;
```

**`example/21_gemm_layernorm/gemm_layernorm_xdl_fp16.cpp`**
```
using CShuffleDataType         = F32;
using DsDataType               = ck::Tuple<>;
using EDataType                = F16;
using R0DataType               = F32;
```
