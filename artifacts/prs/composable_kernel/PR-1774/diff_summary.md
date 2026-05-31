# Diff summary

- **files changed:** 11
- **lines:** +538 / -12
- **kernel-ish files:** 10

## Files (by churn)

- `library/src/tensor_operation_instance/gpu/gemm_multiply_multiply/device_gemm_multiply_multiply_xdl_f8_f8_f16/device_gemm_multiply_multiply_xdl_f8_f8_f16_mk_nk_mn.hpp`  (+153/-0)
- `library/include/ck/library/tensor_operation_instance/gpu/gemm_multiply_multiply.hpp`  (+109/-2)
- `library/src/tensor_operation_instance/gpu/gemm_multiply_multiply/device_gemm_multiply_multiply_xdl_f8_f8_bf16/device_gemm_multiply_multiply_xdl_f8_f8_bf16_mk_nk_mn.hpp`  (+62/-9)
- `library/src/tensor_operation_instance/gpu/gemm_multiply_multiply/device_gemm_multiply_multiply_xdl_f8_f8_f16/device_gemm_multiply_multiply_xdl_f8_f8_f16_mk_nk_mn_mem_v1_default_instance.cpp`  (+33/-0)
- `library/src/tensor_operation_instance/gpu/gemm_multiply_multiply/device_gemm_multiply_multiply_xdl_f8_f8_f16/device_gemm_multiply_multiply_xdl_f8_f8_f16_mk_nk_mn_mem_v1_kpadding_instance.cpp`  (+33/-0)
- `library/src/tensor_operation_instance/gpu/gemm_multiply_multiply/device_gemm_multiply_multiply_xdl_f8_f8_f16/device_gemm_multiply_multiply_xdl_f8_f8_f16_mk_nk_mn_mem_v2_default_instance.cpp`  (+33/-0)
- `library/src/tensor_operation_instance/gpu/gemm_multiply_multiply/device_gemm_multiply_multiply_xdl_f8_f8_f16/device_gemm_multiply_multiply_xdl_f8_f8_f16_mk_nk_mn_mem_v2_kpadding_instance.cpp`  (+33/-0)
- `library/src/tensor_operation_instance/gpu/gemm_multiply_multiply/device_gemm_multiply_multiply_xdl_f8_f8_f16/device_gemm_multiply_multiply_xdl_f8_f8_f16_mk_nk_mn_comp_default_instance.cpp`  (+32/-0)
- `library/src/tensor_operation_instance/gpu/gemm_multiply_multiply/device_gemm_multiply_multiply_xdl_f8_f8_f16/device_gemm_multiply_multiply_xdl_f8_f8_f16_mk_nk_mn_comp_kpadding_instance.cpp`  (+32/-0)
- `library/src/tensor_operation_instance/gpu/gemm_multiply_multiply/CMakeLists.txt`  (+10/-0)
- `profiler/src/profile_gemm_multiply_multiply.cpp`  (+8/-1)

## Key added lines (kernel files)

**`library/include/ck/library/tensor_operation_instance/gpu/gemm_multiply_multiply.hpp`**
```
void add_device_gemm_multiply_multiply_xdl_f8_f8_f16_mk_nk_mn_comp_default_instances(
std::vector<std::unique_ptr<DeviceGemmMultipleDSplitK<Row,
Tuple<Row, Col>,
Tuple<F32, F32>,
```

**`library/src/tensor_operation_instance/gpu/gemm_multiply_multiply/device_gemm_multiply_multiply_xdl_f8_f8_bf16/device_gemm_multiply_multiply_xdl_f8_f8_bf16_mk_nk_mn.hpp`**
```
DeviceGemmMultiD_Xdl_CShuffle_V3<  Row,     Col,     Tuple<Row, Col>,  Row,    F8,    F8,    Tuple<F32, F32>, BF16,  F32
DeviceGemmMultiD_Xdl_CShuffle_V3<  Row,     Col,     Tuple<Row, Col>,  Row,    F8,    F8,    Tuple<F32, F32>, BF16,  F32
DeviceGemmMultiD_Xdl_CShuffle_V3<  Row,     Col,     Tuple<Row, Col>,  Row,    F8,    F8,    Tuple<F32, F32>, BF16,  F32
DeviceGemmMultiD_Xdl_CShuffle_V3<  Row,     Col,     Tuple<Row, Col>,  Row,    F8,    F8,    Tuple<F32, F32>, BF16,  F32
```

**`library/src/tensor_operation_instance/gpu/gemm_multiply_multiply/device_gemm_multiply_multiply_xdl_f8_f8_f16/device_gemm_multiply_multiply_xdl_f8_f8_f16_mk_nk_mn.hpp`**
```
namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {
```

**`library/src/tensor_operation_instance/gpu/gemm_multiply_multiply/device_gemm_multiply_multiply_xdl_f8_f8_f16/device_gemm_multiply_multiply_xdl_f8_f8_f16_mk_nk_mn_comp_default_instance.cpp`**
```
namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {
```

**`library/src/tensor_operation_instance/gpu/gemm_multiply_multiply/device_gemm_multiply_multiply_xdl_f8_f8_f16/device_gemm_multiply_multiply_xdl_f8_f8_f16_mk_nk_mn_comp_kpadding_instance.cpp`**
```
namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {
```
