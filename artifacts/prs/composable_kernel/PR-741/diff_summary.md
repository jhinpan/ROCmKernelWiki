# Diff summary

- **files changed:** 16
- **lines:** +681 / -0
- **kernel-ish files:** 15

## Files (by churn)

- `client_example/02_gemm_add_add_fastgelu/gemm_add_add_fastgelu_generic.cpp`  (+176/-0)
- `client_example/02_gemm_add_add_fastgelu/gemm_add_fastgelu_generic.cpp`  (+169/-0)
- `client_example/02_gemm_add_add_fastgelu/gemm_fastgelu_generic.cpp`  (+162/-0)
- `client_example/02_gemm_add_add_fastgelu/CMakeLists.txt`  (+14/-0)
- `library/src/tensor_operation_instance/gpu/gemm_add_add_fastgelu/device_gemm_add_add_fastgelu_xdl_c_shuffle_f16_f16_f16_f16_f16_km_kn_mn_mn_mn_instance.cpp`  (+14/-0)
- `library/src/tensor_operation_instance/gpu/gemm_add_add_fastgelu/device_gemm_add_add_fastgelu_xdl_c_shuffle_f16_f16_f16_f16_f16_km_nk_mn_mn_mn_instance.cpp`  (+14/-0)
- `library/src/tensor_operation_instance/gpu/gemm_add_add_fastgelu/device_gemm_add_add_fastgelu_xdl_c_shuffle_f16_f16_f16_f16_f16_mk_kn_mn_mn_mn_instance.cpp`  (+14/-0)
- `library/src/tensor_operation_instance/gpu/gemm_add_add_fastgelu/device_gemm_add_add_fastgelu_xdl_c_shuffle_f16_f16_f16_f16_f16_mk_nk_mn_mn_mn_instance.cpp`  (+14/-0)
- `library/src/tensor_operation_instance/gpu/gemm_add_fastgelu/device_gemm_add_fastgelu_xdl_c_shuffle_f16_f16_f16_f16_km_kn_mn_mn_instance.cpp`  (+14/-0)
- `library/src/tensor_operation_instance/gpu/gemm_add_fastgelu/device_gemm_add_fastgelu_xdl_c_shuffle_f16_f16_f16_f16_km_nk_mn_mn_instance.cpp`  (+14/-0)
- `library/src/tensor_operation_instance/gpu/gemm_add_fastgelu/device_gemm_add_fastgelu_xdl_c_shuffle_f16_f16_f16_f16_mk_kn_mn_mn_instance.cpp`  (+14/-0)
- `library/src/tensor_operation_instance/gpu/gemm_add_fastgelu/device_gemm_add_fastgelu_xdl_c_shuffle_f16_f16_f16_f16_mk_nk_mn_mn_instance.cpp`  (+14/-0)
- `library/src/tensor_operation_instance/gpu/gemm_fastgelu/device_gemm_fastgelu_xdl_c_shuffle_f16_f16_f16_km_kn_mn_instance.cpp`  (+12/-0)
- `library/src/tensor_operation_instance/gpu/gemm_fastgelu/device_gemm_fastgelu_xdl_c_shuffle_f16_f16_f16_km_nk_mn_instance.cpp`  (+12/-0)
- `library/src/tensor_operation_instance/gpu/gemm_fastgelu/device_gemm_fastgelu_xdl_c_shuffle_f16_f16_f16_mk_kn_mn_instance.cpp`  (+12/-0)

## Key added lines (kernel files)

**`client_example/02_gemm_add_add_fastgelu/gemm_add_add_fastgelu_generic.cpp`**
```
using F16 = ck::half_t;
using F32 = float;
using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;
```

**`client_example/02_gemm_add_add_fastgelu/gemm_add_fastgelu_generic.cpp`**
```
using F16 = ck::half_t;
using F32 = float;
using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;
```

**`client_example/02_gemm_add_add_fastgelu/gemm_fastgelu_generic.cpp`**
```
using F16 = ck::half_t;
using F32 = float;
using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;
```

**`library/src/tensor_operation_instance/gpu/gemm_add_add_fastgelu/device_gemm_add_add_fastgelu_xdl_c_shuffle_f16_f16_f16_f16_f16_km_kn_mn_mn_mn_instance.cpp`**
```
using device_gemm_add_add_fastgelu_xdl_c_shuffle_f16_f16_f16_f16_f16_km_kn_mn_mn_mn_generic_instance =
std::tuple<
DeviceGemmMultipleD_Xdl_CShuffle<    Col,    Row, Row_Row_Tuple,    Row,   F16,   F16,     F32,      F32, F16_F16_Tuple,
add_device_operation_instances(
```

**`library/src/tensor_operation_instance/gpu/gemm_add_add_fastgelu/device_gemm_add_add_fastgelu_xdl_c_shuffle_f16_f16_f16_f16_f16_km_nk_mn_mn_mn_instance.cpp`**
```
using device_gemm_add_add_fastgelu_xdl_c_shuffle_f16_f16_f16_f16_f16_km_nk_mn_mn_mn_generic_instance =
std::tuple<
DeviceGemmMultipleD_Xdl_CShuffle<    Col,    Col, Row_Row_Tuple,    Row,   F16,   F16,     F32,      F32, F16_F16_Tuple,
add_device_operation_instances(
```
