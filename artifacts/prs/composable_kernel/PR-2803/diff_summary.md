# Diff summary

- **files changed:** 25 (diff was byte-capped; summary is partial)
- **lines:** +3989 / -312
- **kernel-ish files:** 23

## Files (by churn)

- `library/include/ck/library/tensor_operation_instance/gpu/gemm_multi_abd.hpp`  (+948/-6)
- `include/ck/tensor_operation/gpu/device/impl/device_gemm_multiple_abd_wmma_cshuffle_v3.hpp`  (+422/-0)
- `include/ck/tensor_operation/gpu/grid/gridwise_gemm_wmma_cshuffle_v3_common.hpp`  (+278/-108)
- `example/60_gemm_multi_ABD/gemm_multi_ABD_wmma_fp16.cpp`  (+362/-0)
- `example/60_gemm_multi_ABD/gemm_multi_ABD_wmma_bias_fastgelu_bf16_i8.cpp`  (+307/-0)
- `example/60_gemm_multi_ABD/gemm_multi_ABD_wmma_fastgelu_bf16_i8.cpp`  (+299/-0)
- `example/60_gemm_multi_ABD/gemm_multi_ABD_wmma_multiply_bias_fastgelu_bf16_i8.cpp`  (+296/-0)
- `include/ck/tensor_operation/gpu/grid/gridwise_gemm_wmma_cshuffle_v3_b_scale.hpp`  (+110/-49)
- `include/ck/tensor_operation/gpu/grid/gridwise_gemm_wmma_cshuffle_v3.hpp`  (+107/-48)
- `include/ck/tensor_operation/gpu/device/device_gemm_multiple_abd.hpp`  (+150/-1)
- `include/ck/host_utility/flush_cache.hpp`  (+148/-1)
- `library/src/tensor_operation_instance/gpu/gemm_multi_abd/device_gemm_wmma_multi_abd_bf16_i8_bf16_mk_kn_mn_common.hpp`  (+109/-0)
- `library/src/tensor_operation_instance/gpu/gemm_multi_abd/device_gemm_wmma_multi_abd_bf16_i8_bf16_mk_nk_mn_common.hpp`  (+85/-0)
- `include/ck/tensor_operation/gpu/device/impl/device_batched_gemm_wmma_cshuffle_v3.hpp`  (+47/-22)
- `library/src/tensor_operation_instance/gpu/gemm_multi_abd/device_gemm_wmma_multi_abd_bf16_i8_bf16_mk_kn_mn_v1_instance.cpp`  (+58/-0)

## Key added lines (kernel files)

**`example/60_gemm_multi_ABD/gemm_multi_ABD_wmma_bias_fastgelu_bf16_i8.cpp`**
```
template <ck::index_t... Is>
using S = ck::Sequence<Is...>;
using BF16 = ck::bhalf_t;
using I8   = int8_t;
```

**`example/60_gemm_multi_ABD/gemm_multi_ABD_wmma_fastgelu_bf16_i8.cpp`**
```
template <ck::index_t... Is>
using S = ck::Sequence<Is...>;
using BF16 = ck::bhalf_t;
using I8   = int8_t;
```

**`example/60_gemm_multi_ABD/gemm_multi_ABD_wmma_fp16.cpp`**
```
template <ck::index_t... Is>
using S = ck::Sequence<Is...>;
using F16 = ck::half_t;
using F32 = float;
```

**`example/60_gemm_multi_ABD/gemm_multi_ABD_wmma_multiply_bias_fastgelu_bf16_i8.cpp`**
```
template <ck::index_t... Is>
using S = ck::Sequence<Is...>;
using BF16 = ck::bhalf_t;
using I8   = int8_t;
```

**`include/ck/host_utility/flush_cache.hpp`**
```
template <typename Argument, typename AsDataType, typename BsDataType, typename DsDataType>
struct RotatingMemWrapperMultiABD
static constexpr index_t NumAs = AsDataType::Size();
static constexpr index_t NumBs = BsDataType::Size();
```
