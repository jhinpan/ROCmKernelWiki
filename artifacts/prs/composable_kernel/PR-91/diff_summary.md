# Diff summary

- **files changed:** 16
- **lines:** +1203 / -74
- **kernel-ish files:** 14

## Files (by churn)

- `profiler/include/profile_gemm_bias_2d_impl.hpp`  (+311/-0)
- `profiler/src/profile_gemm_bias_2d.cpp`  (+261/-0)
- `reference_operation/include/reference_gemm_bias_2d.hpp`  (+133/-0)
- `device_operation/CMakeLists.txt`  (+45/-28)
- `example/8_gemm_xdl_alpha_beta/gemm_xdl_alpha_beta.cpp`  (+21/-44)
- `device_operation/src/device_gemm_xdl_c_shuffle_bias_2d_f16_f16_f16_mk_nk_mn_instance.cpp`  (+57/-0)
- `device_operation/src/device_gemm_xdl_c_shuffle_bias_2d_f32_f32_f32_mk_nk_mn_instance.cpp`  (+56/-0)
- `device_operation/src/device_gemm_xdl_c_shuffle_bias_2d_f16_f16_f16_km_kn_mn_instance.cpp`  (+52/-0)
- `device_operation/src/device_gemm_xdl_c_shuffle_bias_2d_f16_f16_f16_km_nk_mn_instance.cpp`  (+52/-0)
- `device_operation/src/device_gemm_xdl_c_shuffle_bias_2d_f16_f16_f16_mk_kn_mn_instance.cpp`  (+52/-0)
- `device_operation/src/device_gemm_xdl_c_shuffle_bias_2d_f32_f32_f32_km_kn_mn_instance.cpp`  (+51/-0)
- `device_operation/src/device_gemm_xdl_c_shuffle_bias_2d_f32_f32_f32_km_nk_mn_instance.cpp`  (+51/-0)
- `device_operation/src/device_gemm_xdl_c_shuffle_bias_2d_f32_f32_f32_mk_kn_mn_instance.cpp`  (+51/-0)
- `profiler/src/profiler.cpp`  (+6/-0)
- `profiler/CMakeLists.txt`  (+3/-1)

## Key added lines (kernel files)

**`device_operation/include/device_gemm_xdl_c_shuffle_bias_2d.hpp`**
```
str << "DeviceGemmXdl_C_Shuffle_Bias_2d"
```

**`device_operation/src/device_gemm_xdl_c_shuffle_bias_2d_f16_f16_f16_km_kn_mn_instance.cpp`**
```
namespace ck {
namespace tensor_operation {
namespace device {
namespace device_gemm_instance {
```

**`device_operation/src/device_gemm_xdl_c_shuffle_bias_2d_f16_f16_f16_km_nk_mn_instance.cpp`**
```
namespace ck {
namespace tensor_operation {
namespace device {
namespace device_gemm_instance {
```

**`device_operation/src/device_gemm_xdl_c_shuffle_bias_2d_f16_f16_f16_mk_kn_mn_instance.cpp`**
```
namespace ck {
namespace tensor_operation {
namespace device {
namespace device_gemm_instance {
```

**`device_operation/src/device_gemm_xdl_c_shuffle_bias_2d_f16_f16_f16_mk_nk_mn_instance.cpp`**
```
namespace ck {
namespace tensor_operation {
namespace device {
namespace device_gemm_instance {
```
