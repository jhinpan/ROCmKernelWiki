# Diff summary

- **files changed:** 24
- **lines:** +665 / -399
- **kernel-ish files:** 22

## Files (by churn)

- `profiler/include/profiler/profile_grouped_gemm_fastgelu_impl.hpp`  (+22/-232)
- `profiler/include/profiler/profile_grouped_gemm_impl.hpp`  (+83/-47)
- `library/include/ck/library/tensor_operation_instance/gpu/grouped_gemm/device_grouped_gemm_wmma_splitk_instance.hpp`  (+67/-32)
- `test/grouped_gemm/test_grouped_gemm_util.hpp`  (+72/-26)
- `include/ck/tensor_operation/gpu/device/impl/device_grouped_gemm_wmma_splitk_cshuffle_v3.hpp`  (+72/-23)
- `library/include/ck/library/tensor_operation_instance/gpu/grouped_gemm_fastgelu.hpp`  (+82/-0)
- `test/grouped_gemm/test_grouped_gemm_fastgelu.cpp`  (+62/-0)
- `library/src/tensor_operation_instance/gpu/grouped_gemm_fastgelu/device_grouped_gemm_fastgelu_wmma_f16_f16_f16_mk_kn_mn_instance.cpp`  (+38/-0)
- `library/src/tensor_operation_instance/gpu/grouped_gemm_fastgelu/device_grouped_gemm_fastgelu_wmma_f16_f16_f16_mk_nk_mn_instance.cpp`  (+38/-0)
- `library/src/tensor_operation_instance/gpu/grouped_gemm_fastgelu/device_grouped_gemm_fastgelu_wmma_f16_f16_f16_km_kn_mn_instance.cpp`  (+37/-0)
- `library/src/tensor_operation_instance/gpu/grouped_gemm_fastgelu/device_grouped_gemm_fastgelu_wmma_f16_f16_f16_km_nk_mn_instance.cpp`  (+37/-0)
- `library/src/tensor_operation_instance/gpu/grouped_gemm/device_grouped_gemm_wmma_universal_f16_f8_f16_mk_kn_mn_instance.cpp`  (+7/-4)
- `library/src/tensor_operation_instance/gpu/grouped_gemm/device_grouped_gemm_wmma_universal_f8_f16_f16_mk_kn_mn_instance.cpp`  (+7/-4)
- `test/grouped_gemm/test_grouped_gemm_ut_cases.inc`  (+5/-6)
- `library/src/tensor_operation_instance/gpu/grouped_gemm_fastgelu/CMakeLists.txt`  (+6/-1)

## Key added lines (kernel files)

**`include/ck/tensor_operation/gpu/device/impl/device_grouped_gemm_wmma_splitk_cshuffle_v3.hpp`**
```
std::vector<std::array<const void*, NumDTensor>>& p_Ds,
std::vector<GemmDesc>& gemm_descs,
AElementwiseOperation a_element_op,
BElementwiseOperation b_element_op,
```

**`library/include/ck/library/tensor_operation_instance/gpu/grouped_gemm/device_grouped_gemm_wmma_splitk_instance.hpp`**
```
using FastGelu    = ck::tensor_operation::element_wise::FastGelu;
typename AElementOp,
typename BElementOp,
typename CDEElementOp,
```

**`library/include/ck/library/tensor_operation_instance/gpu/grouped_gemm_fastgelu.hpp`**
```
void add_device_grouped_gemm_fastgelu_wmma_f16_f16_f16_mk_kn_mn_instances(
std::vector<std::unique_ptr<DeviceGroupedGemm<Row,
Empty_Tuple,
Empty_Tuple,
```

**`library/src/tensor_operation_instance/gpu/grouped_gemm/device_grouped_gemm_wmma_universal_bf16_bf16_bf16_km_kn_mn_instance.cpp`**
```
PassThrough,
PassThrough,
PassThrough>>>& instances)
```

**`library/src/tensor_operation_instance/gpu/grouped_gemm/device_grouped_gemm_wmma_universal_bf16_bf16_bf16_km_nk_mn_instance.cpp`**
```
PassThrough,
PassThrough,
PassThrough>>>& instances)
```
