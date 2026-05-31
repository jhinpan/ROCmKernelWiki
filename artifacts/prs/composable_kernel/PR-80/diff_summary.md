# Diff summary

- **files changed:** 15
- **lines:** +352 / -200
- **kernel-ish files:** 15

## Files (by churn)

- `device_operation/include/device_gemm_xdl.hpp`  (+69/-20)
- `device_operation/include/device_gemm_xdl_splitk.hpp`  (+68/-20)
- `device_operation/src/device_gemm_xdl_f16_f16_f16_mk_nk_mn_instance.cpp`  (+19/-17)
- `device_operation/src/device_gemm_xdl_f32_f32_f32_mk_nk_mn_instance.cpp`  (+19/-17)
- `device_operation/src/device_gemm_xdl_splitk_f32_f32_f32_mk_nk_mn_instance.cpp`  (+19/-17)
- `device_operation/src/device_gemm_xdl_f16_f16_f16_mk_kn_mn_instance.cpp`  (+23/-12)
- `device_operation/src/device_gemm_xdl_splitk_f32_f32_f32_mk_kn_mn_instance.cpp`  (+20/-13)
- `device_operation/src/device_gemm_xdl_f16_f16_f16_km_kn_mn_instance.cpp`  (+14/-12)
- `device_operation/src/device_gemm_xdl_f16_f16_f16_km_nk_mn_instance.cpp`  (+14/-12)
- `device_operation/src/device_gemm_xdl_f32_f32_f32_km_kn_mn_instance.cpp`  (+14/-12)
- `device_operation/src/device_gemm_xdl_f32_f32_f32_km_nk_mn_instance.cpp`  (+14/-12)
- `device_operation/src/device_gemm_xdl_f32_f32_f32_mk_kn_mn_instance.cpp`  (+14/-12)
- `device_operation/src/device_gemm_xdl_splitk_f32_f32_f32_km_kn_mn_instance.cpp`  (+14/-12)
- `device_operation/src/device_gemm_xdl_splitk_f32_f32_f32_km_nk_mn_instance.cpp`  (+14/-12)
- `device_operation/include/gemm_specialization.hpp`  (+17/-0)

## Key added lines (kernel files)

**`device_operation/include/device_gemm_xdl.hpp`**
```
GemmSpecialization_t GemmSpecialization,
if constexpr(GemmSpecialization == GemmSpecialization_t::MNPadding)
const auto PadM = (MPerBlock - M % MPerBlock) % MPerBlock;
return transform_tensor_descriptor(
```

**`device_operation/include/device_gemm_xdl_splitk.hpp`**
```
GemmSpecialization_t GemmSpecialization,
if constexpr(GemmSpecialization == GemmSpecialization_t::MNPadding)
const auto PadM = (MPerBlock - M % MPerBlock) % MPerBlock;
return transform_tensor_descriptor(
```

**`device_operation/include/gemm_specialization.hpp`**
```
namespace ck {
namespace tensor_operation {
namespace device {
enum GemmSpecialization_t
```

**`device_operation/src/device_gemm_xdl_f16_f16_f16_km_kn_mn_instance.cpp`**
```
static constexpr auto GemmDefault = ck::tensor_operation::device::GemmSpecialization_t::Default;
DeviceGemmXdl<  F16,   F16,   F16,     F32,     Col,     Row,     Row, PassThrough, PassThrough, PassThrough,   GemmDefa
DeviceGemmXdl<  F16,   F16,   F16,     F32,     Col,     Row,     Row, PassThrough, PassThrough, PassThrough,   GemmDefa
DeviceGemmXdl<  F16,   F16,   F16,     F32,     Col,     Row,     Row, PassThrough, PassThrough, PassThrough,   GemmDefa
```

**`device_operation/src/device_gemm_xdl_f16_f16_f16_km_nk_mn_instance.cpp`**
```
static constexpr auto GemmDefault = ck::tensor_operation::device::GemmSpecialization_t::Default;
DeviceGemmXdl<  F16,   F16,   F16,     F32,     Col,      Col,    Row, PassThrough, PassThrough, PassThrough,   GemmDefa
DeviceGemmXdl<  F16,   F16,   F16,     F32,     Col,      Col,    Row, PassThrough, PassThrough, PassThrough,   GemmDefa
DeviceGemmXdl<  F16,   F16,   F16,     F32,     Col,      Col,    Row, PassThrough, PassThrough, PassThrough,   GemmDefa
```
