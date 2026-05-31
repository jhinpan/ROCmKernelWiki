# Diff summary

- **files changed:** 11
- **lines:** +562 / -23
- **kernel-ish files:** 10

## Files (by churn)

- `include/ck/tensor_operation/gpu/warp/dpp_gemm.hpp`  (+223/-7)
- `library/src/tensor_operation_instance/gpu/gemm/device_gemm_dpp_f16_f16_f16_km_kn_mn_irregular_instance.cpp`  (+64/-0)
- `library/src/tensor_operation_instance/gpu/gemm/device_gemm_dpp_f16_f16_f16_km_nk_mn_irregular_instance.cpp`  (+64/-0)
- `library/src/tensor_operation_instance/gpu/gemm/device_gemm_dpp_f16_f16_f16_mk_nk_mn_irregular_instance.cpp`  (+64/-0)
- `library/src/tensor_operation_instance/gpu/gemm/device_gemm_dpp_f16_f16_f16_mk_kn_mn_irregular_instance.cpp`  (+63/-0)
- `library/include/ck/library/tensor_operation_instance/gpu/gemm.hpp`  (+28/-4)
- `library/src/tensor_operation_instance/gpu/gemm/device_gemm_dpp_f16_f16_f16_km_kn_mn_instance.cpp`  (+13/-3)
- `library/src/tensor_operation_instance/gpu/gemm/device_gemm_dpp_f16_f16_f16_km_nk_mn_instance.cpp`  (+13/-3)
- `library/src/tensor_operation_instance/gpu/gemm/device_gemm_dpp_f16_f16_f16_mk_kn_mn_instance.cpp`  (+13/-3)
- `library/src/tensor_operation_instance/gpu/gemm/device_gemm_dpp_f16_f16_f16_mk_nk_mn_instance.cpp`  (+13/-3)
- `library/src/tensor_operation_instance/gpu/gemm/CMakeLists.txt`  (+4/-0)

## Key added lines (kernel files)

**`include/ck/tensor_operation/gpu/warp/dpp_gemm.hpp`**
```
dpp8_f16_1x32x2 = 0,
dpp8_f16_2x16x2,
dpp8_f16_2x32x2,
dpp8_f16_4x16x2,
```

**`library/include/ck/library/tensor_operation_instance/gpu/gemm.hpp`**
```
void add_device_gemm_dl_f16_f16_f16_km_kn_mn_irregular_instances(
std::vector<std::unique_ptr<
DeviceGemm<Col, Row, Row, F16, F16, F16, PassThrough, PassThrough, PassThrough>>>&
instances);
```

**`library/src/tensor_operation_instance/gpu/gemm/device_gemm_dpp_f16_f16_f16_km_kn_mn_instance.cpp`**
```
DeviceGemmDpp< F16,   F16,   F16,     F32,     Col,     Row,     Row, PassThrough, PassThrough, PassThrough,    GemmDefa
DeviceGemmDpp< F16,   F16,   F16,     F32,     Col,     Row,     Row, PassThrough, PassThrough, PassThrough,    GemmDefa
DeviceGemmDpp< F16,   F16,   F16,     F32,     Col,     Row,     Row, PassThrough, PassThrough, PassThrough,    GemmDefa
DeviceGemmDpp< F16,   F16,   F16,     F32,     Col,     Row,     Row, PassThrough, PassThrough, PassThrough,    GemmDefa
```

**`library/src/tensor_operation_instance/gpu/gemm/device_gemm_dpp_f16_f16_f16_km_kn_mn_irregular_instance.cpp`**
```
namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {
```

**`library/src/tensor_operation_instance/gpu/gemm/device_gemm_dpp_f16_f16_f16_km_nk_mn_instance.cpp`**
```
DeviceGemmDpp< F16,   F16,   F16,     F32,     Col,     Col,     Row, PassThrough, PassThrough, PassThrough,    GemmDefa
DeviceGemmDpp< F16,   F16,   F16,     F32,     Col,     Col,     Row, PassThrough, PassThrough, PassThrough,    GemmDefa
DeviceGemmDpp< F16,   F16,   F16,     F32,     Col,     Col,     Row, PassThrough, PassThrough, PassThrough,    GemmDefa
DeviceGemmDpp< F16,   F16,   F16,     F32,     Col,     Col,     Row, PassThrough, PassThrough, PassThrough,    GemmDefa
```
