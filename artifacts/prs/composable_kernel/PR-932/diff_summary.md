# Diff summary

- **files changed:** 98
- **lines:** +1281 / -1007
- **kernel-ish files:** 13

## Files (by churn)

- `library/src/tensor_operation_instance/gpu/gemm/CMakeLists.txt`  (+93/-107)
- `library/include/ck/library/tensor_operation_instance/gpu/grouped_convolution_forward.hpp`  (+123/-52)
- `library/include/ck/library/tensor_operation_instance/gpu/grouped_convolution_backward_weight.hpp`  (+102/-31)
- `test/CMakeLists.txt`  (+108/-13)
- `example/CMakeLists.txt`  (+104/-10)
- `library/src/tensor_operation_instance/gpu/CMakeLists.txt`  (+83/-19)
- `example/01_gemm/CMakeLists.txt`  (+43/-39)
- `library/include/ck/library/tensor_operation_instance/gpu/grouped_convolution_backward_data.hpp`  (+48/-12)
- `library/include/ck/library/tensor_operation_instance/gpu/reduce/device_reduce_instance.hpp`  (+29/-30)
- `example/15_grouped_gemm/CMakeLists.txt`  (+33/-25)
- `library/include/ck/library/tensor_operation_instance/gpu/batchnorm_backward.hpp`  (+28/-22)
- `library/include/ck/library/tensor_operation_instance/gpu/batchnorm_forward.hpp`  (+25/-19)
- `library/include/ck/library/tensor_operation_instance/gpu/batchnorm_infer.hpp`  (+25/-19)
- `library/src/tensor_operation_instance/gpu/batched_gemm_multi_d/CMakeLists.txt`  (+20/-24)
- `example/16_gemm_multi_d_multi_reduces/CMakeLists.txt`  (+29/-14)

## Key added lines (kernel files)

**`library/include/ck/library/tensor_operation_instance/gpu/batchnorm_backward.hpp`**
```
if constexpr(is_same_v<XDataType, F32> && is_same_v<DxDataType, F32> &&
is_same_v<DyDataType, F32> && is_same_v<AccDataType, F32> &&
is_same_v<ScaleDataType, F32> && is_same_v<DscaleDbiasDataType, F32> &&
is_same_v<MeanVarDataType, F32>)
```

**`library/include/ck/library/tensor_operation_instance/gpu/batchnorm_forward.hpp`**
```
if constexpr(is_same_v<XDataType, F32> && is_same_v<YDataType, F32> &&
is_same_v<AccDataType, F32> && is_same_v<ScaleDataType, F32> &&
is_same_v<BiasDataType, F32> && is_same_v<MeanVarDataType, F32>)
if constexpr(is_same_v<XDataType, BF16> && is_same_v<YDataType, BF16> &&
```

**`library/include/ck/library/tensor_operation_instance/gpu/batchnorm_infer.hpp`**
```
if constexpr(is_same_v<XDataType, F32> && is_same_v<YDataType, F32> &&
is_same_v<ScaleDataType, F32> && is_same_v<BiasDataType, F32> &&
is_same_v<MeanVarDataType, F32>)
if constexpr(is_same_v<XDataType, BF16> && is_same_v<YDataType, BF16> &&
```

**`library/include/ck/library/tensor_operation_instance/gpu/grouped_convolution_forward.hpp`**
```
if constexpr(is_same_v<InDataType, half_t> && is_same_v<WeiDataType, half_t> &&
is_same_v<OutDataType, half_t>)
if constexpr(is_same_v<InDataType, ck::bhalf_t> &&
is_same_v<WeiDataType, ck::bhalf_t> && is_same_v<OutDataType, ck::bhalf_t>)
```
