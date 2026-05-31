# Diff summary

- **files changed:** 17
- **lines:** +1320 / -26
- **kernel-ish files:** 14

## Files (by churn)

- `profiler/include/profiler/profile_grouped_gemm_fastgelu_impl.hpp`  (+280/-0)
- `client_example/17_grouped_gemm_fastgelu/grouped_gemm_fastgelu.cpp`  (+232/-0)
- `profiler/src/profile_grouped_gemm_fastgelu.cpp`  (+177/-0)
- `library/include/ck/library/tensor_operation_instance/gpu/grouped_gemm_fastgelu.hpp`  (+136/-0)
- `library/src/tensor_operation_instance/gpu/grouped_gemm_fastgelu/device_grouped_gemm_fastgelu_xdl_f16_f16_f16_km_kn_mn_instance.cpp`  (+104/-0)
- `library/src/tensor_operation_instance/gpu/grouped_gemm_fastgelu/device_grouped_gemm_fastgelu_xdl_f16_f16_f16_km_nk_mn_instance.cpp`  (+104/-0)
- `library/src/tensor_operation_instance/gpu/grouped_gemm_fastgelu/device_grouped_gemm_fastgelu_xdl_f16_f16_f16_mk_nk_mn_instance.cpp`  (+104/-0)
- `library/src/tensor_operation_instance/gpu/grouped_gemm_fastgelu/device_grouped_gemm_fastgelu_xdl_f16_f16_f16_mk_kn_mn_instance.cpp`  (+103/-0)
- `include/ck/tensor_operation/gpu/device/impl/device_grouped_gemm_xdl.hpp`  (+29/-2)
- `library/include/ck/library/tensor_operation_instance/gpu/grouped_gemm.hpp`  (+14/-15)
- `library/include/ck/library/utility/fill.hpp`  (+18/-0)
- `profiler/src/profile_grouped_gemm.cpp`  (+7/-3)
- `profiler/include/profiler/profile_grouped_gemm_impl.hpp`  (+1/-6)
- `library/src/tensor_operation_instance/gpu/grouped_gemm_fastgelu/CMakeLists.txt`  (+6/-0)
- `client_example/17_grouped_gemm_fastgelu/CMakeLists.txt`  (+2/-0)

## Key added lines (kernel files)

**`client_example/17_grouped_gemm_fastgelu/grouped_gemm_fastgelu.cpp`**
```
using F16 = ck::half_t;
using F32 = float;
using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;
```

**`include/ck/tensor_operation/gpu/device/impl/device_grouped_gemm_xdl.hpp`**
```
a_mtx_mraw_kraw_.emplace_back(M, K);
b_mtx_nraw_kraw_.emplace_back(N, K);
std::vector<Tuple<index_t, index_t>> a_mtx_mraw_kraw_;
std::vector<Tuple<index_t, index_t>> b_mtx_nraw_kraw_;
```

**`library/include/ck/library/tensor_operation_instance/device_operation_instance_factory.hpp`**
```
using Gelu           = ck::tensor_operation::element_wise::Gelu;
```

**`library/include/ck/library/tensor_operation_instance/gpu/grouped_gemm.hpp`**
```
struct DeviceOperationInstanceFactory<ck::tensor_operation::device::DeviceGroupedGemm<ALayout,
Empty_Tuple,
ADataType,
BDataType,
```

**`library/include/ck/library/tensor_operation_instance/gpu/grouped_gemm_fastgelu.hpp`**
```
namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {
```
