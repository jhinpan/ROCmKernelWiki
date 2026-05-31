# Diff summary

- **files changed:** 18
- **lines:** +937 / -8
- **kernel-ish files:** 17

## Files (by churn)

- `library/src/tensor_operation_instance/gpu/gemm/device_gemm_dl_i8_i8_i8_km_kn_mn_irregular_instance.cpp`  (+80/-0)
- `library/src/tensor_operation_instance/gpu/gemm/device_gemm_dl_i8_i8_i8_km_nk_mn_irregular_instance.cpp`  (+80/-0)
- `library/src/tensor_operation_instance/gpu/gemm/device_gemm_dl_i8_i8_i8_mk_kn_mn_irregular_instance.cpp`  (+80/-0)
- `library/src/tensor_operation_instance/gpu/gemm/device_gemm_dl_i8_i8_i8_mk_nk_mn_irregular_instance.cpp`  (+80/-0)
- `library/src/tensor_operation_instance/gpu/gemm/device_gemm_dl_f16_f16_f16_mk_nk_mn_irregular_instance.cpp`  (+74/-0)
- `library/src/tensor_operation_instance/gpu/gemm/device_gemm_dl_f16_f16_f16_km_kn_mn_irregular_instance.cpp`  (+73/-0)
- `library/src/tensor_operation_instance/gpu/gemm/device_gemm_dl_f16_f16_f16_km_nk_mn_irregular_instance.cpp`  (+73/-0)
- `library/src/tensor_operation_instance/gpu/gemm/device_gemm_dl_f16_f16_f16_mk_kn_mn_irregular_instance.cpp`  (+73/-0)
- `library/include/ck/library/tensor_operation_instance/gpu/gemm.hpp`  (+48/-0)
- `library/src/tensor_operation_instance/gpu/gemm/device_gemm_dl_i8_i8_i8_km_kn_mn_instance.cpp`  (+34/-1)
- `library/src/tensor_operation_instance/gpu/gemm/device_gemm_dl_i8_i8_i8_km_nk_mn_instance.cpp`  (+34/-1)
- `library/src/tensor_operation_instance/gpu/gemm/device_gemm_dl_i8_i8_i8_mk_kn_mn_instance.cpp`  (+34/-1)
- `library/src/tensor_operation_instance/gpu/gemm/device_gemm_dl_i8_i8_i8_mk_nk_mn_instance.cpp`  (+34/-1)
- `library/src/tensor_operation_instance/gpu/gemm/device_gemm_dl_f16_f16_f16_km_kn_mn_instance.cpp`  (+33/-1)
- `library/src/tensor_operation_instance/gpu/gemm/device_gemm_dl_f16_f16_f16_km_nk_mn_instance.cpp`  (+33/-1)

## Key added lines (kernel files)

**`library/include/ck/library/tensor_operation_instance/gpu/gemm.hpp`**
```
void add_device_gemm_dl_f16_f16_f16_km_kn_mn_irregular_instances(
std::vector<std::unique_ptr<
DeviceGemm<Col, Row, Row, F16, F16, F16, PassThrough, PassThrough, PassThrough>>>&
instances);
```

**`library/src/tensor_operation_instance/gpu/gemm/device_gemm_dl_f16_f16_f16_km_kn_mn_instance.cpp`**
```
DeviceGemmDl<   F16,   F16,   F16,     F32,     Col,     Row,     Row, PassThrough, PassThrough, PassThrough,    GemmDef
DeviceGemmDl<   F16,   F16,   F16,     F32,     Col,     Row,     Row, PassThrough, PassThrough, PassThrough,    GemmDef
DeviceGemmDl<   F16,   F16,   F16,     F32,     Col,     Row,     Row, PassThrough, PassThrough, PassThrough,    GemmDef
DeviceGemmDl<   F16,   F16,   F16,     F32,     Col,     Row,     Row, PassThrough, PassThrough, PassThrough,    GemmDef
```

**`library/src/tensor_operation_instance/gpu/gemm/device_gemm_dl_f16_f16_f16_km_kn_mn_irregular_instance.cpp`**
```
namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {
```

**`library/src/tensor_operation_instance/gpu/gemm/device_gemm_dl_f16_f16_f16_km_nk_mn_instance.cpp`**
```
DeviceGemmDl<   F16,   F16,   F16,     F32,     Col,     Col,     Row, PassThrough, PassThrough, PassThrough,    GemmDef
DeviceGemmDl<   F16,   F16,   F16,     F32,     Col,     Col,     Row, PassThrough, PassThrough, PassThrough,    GemmDef
DeviceGemmDl<   F16,   F16,   F16,     F32,     Col,     Col,     Row, PassThrough, PassThrough, PassThrough,    GemmDef
DeviceGemmDl<   F16,   F16,   F16,     F32,     Col,     Col,     Row, PassThrough, PassThrough, PassThrough,    GemmDef
```

**`library/src/tensor_operation_instance/gpu/gemm/device_gemm_dl_f16_f16_f16_km_nk_mn_irregular_instance.cpp`**
```
namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {
```
