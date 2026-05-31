# Diff summary

- **files changed:** 16 (diff was byte-capped; summary is partial)
- **lines:** +1512 / -10
- **kernel-ish files:** 13

## Files (by churn)

- `client_example/02_gemm_add_add_fastgelu/gemm_add_fastgelu.cpp`  (+233/-0)
- `client_example/02_gemm_add_add_fastgelu/gemm_fastgelu.cpp`  (+225/-0)
- `library/include/ck/library/tensor_operation_instance/gpu/gemm_add_fastgelu.hpp`  (+145/-0)
- `library/include/ck/library/tensor_operation_instance/gpu/gemm_fastgelu.hpp`  (+138/-0)
- `library/src/tensor_operation_instance/gpu/gemm_add_fastgelu/device_gemm_add_fastgelu_xdl_c_shuffle_f16_f16_f16_f16_km_kn_mn_mn_instance.cpp`  (+109/-0)
- `library/src/tensor_operation_instance/gpu/gemm_add_fastgelu/device_gemm_add_fastgelu_xdl_c_shuffle_f16_f16_f16_f16_km_nk_mn_mn_instance.cpp`  (+109/-0)
- `library/src/tensor_operation_instance/gpu/gemm_add_fastgelu/device_gemm_add_fastgelu_xdl_c_shuffle_f16_f16_f16_f16_mk_kn_mn_mn_instance.cpp`  (+109/-0)
- `library/src/tensor_operation_instance/gpu/gemm_fastgelu/device_gemm_fastgelu_xdl_c_shuffle_f16_f16_f16_km_kn_mn_instance.cpp`  (+109/-0)
- `library/src/tensor_operation_instance/gpu/gemm_fastgelu/device_gemm_fastgelu_xdl_c_shuffle_f16_f16_f16_km_nk_mn_instance.cpp`  (+109/-0)
- `library/src/tensor_operation_instance/gpu/gemm_add_fastgelu/device_gemm_add_fastgelu_xdl_c_shuffle_f16_f16_f16_f16_mk_nk_mn_mn_instance.cpp`  (+100/-0)
- `library/src/tensor_operation_instance/gpu/gemm_fastgelu/device_gemm_fastgelu_xdl_c_shuffle_f16_f16_f16_mk_kn_mn_instance.cpp`  (+76/-0)
- `include/ck/tensor_operation/gpu/element/unary_element_wise_operation.hpp`  (+25/-10)
- `client_example/02_gemm_add_add_fastgelu/CMakeLists.txt`  (+11/-0)
- `library/src/tensor_operation_instance/gpu/gemm_add_fastgelu/CMakeLists.txt`  (+6/-0)
- `library/src/tensor_operation_instance/gpu/gemm_fastgelu/CMakeLists.txt`  (+6/-0)

## Key added lines (kernel files)

**`client_example/02_gemm_add_add_fastgelu/gemm_add_fastgelu.cpp`**
```
using F16 = ck::half_t;
using F32 = float;
using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;
```

**`client_example/02_gemm_add_add_fastgelu/gemm_fastgelu.cpp`**
```
using F16 = ck::half_t;
using F32 = float;
using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;
```

**`include/ck/tensor_operation/gpu/element/unary_element_wise_operation.hpp`**
```
__host__ __device__ static constexpr float GetFastGeLU(float x)
const float u   = 2.f * x * (0.035677f * x * x + 0.797885f);
const float cdf = 0.5f + 0.5f * (2.f / (1.f + emu) - 1.f);
return x * cdf;
```

**`library/include/ck/library/tensor_operation_instance/device_operation_instance_factory.hpp`**
```
using AddFastGelu    = ck::tensor_operation::element_wise::AddFastGelu;
using FastGelu       = ck::tensor_operation::element_wise::FastGelu;
```

**`library/include/ck/library/tensor_operation_instance/gpu/gemm_add_fastgelu.hpp`**
```
namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {
```
