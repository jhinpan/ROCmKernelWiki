# Diff summary

- **files changed:** 41 (diff was byte-capped; summary is partial)
- **lines:** +1411 / -352
- **kernel-ish files:** 40

## Files (by churn)

- `library/src/tensor_operation_instance/gpu/gemm/CMakeLists.txt`  (+105/-41)
- `library/src/tensor_operation_instance/gpu/gemm/device_gemm_xdl_f16_f16_f16_km_kn_mn_instance.cpp`  (+0/-110)
- `library/src/tensor_operation_instance/gpu/gemm/device_gemm_xdl_f16_f16_f16_km_nk_mn_instance.cpp`  (+0/-110)
- `library/src/tensor_operation_instance/gpu/gemm/device_gemm_xdl_f16_f16_f16_mk_kn_mn_instance.cpp`  (+0/-91)
- `library/src/tensor_operation_instance/gpu/gemm/device_gemm_xdl_f16_f16_f16/mk_kn_mn_default_pipeline_v2_instance.cpp`  (+51/-0)
- `library/src/tensor_operation_instance/gpu/gemm/device_gemm_xdl_f16_f16_f16/mk_kn_mn_interwave_pipeline_v1_instance.cpp`  (+51/-0)
- `library/src/tensor_operation_instance/gpu/gemm/device_gemm_xdl_f16_f16_f16/mk_kn_mn_default_pipeline_v1_instance.cpp`  (+49/-0)
- `library/src/tensor_operation_instance/gpu/gemm/device_gemm_xdl_f16_f16_f16/mk_nk_mn_default_pipeline_v2_instance.cpp`  (+46/-0)
- `library/src/tensor_operation_instance/gpu/gemm/device_gemm_xdl_f16_f16_f16/mk_nk_mn_interwave_pipeline_v1_instance.cpp`  (+46/-0)
- `library/src/tensor_operation_instance/gpu/gemm/device_gemm_xdl_f16_f16_f16/mk_nk_mn_default_pipeline_v1_instance.cpp`  (+44/-0)
- `library/src/tensor_operation_instance/gpu/gemm/device_gemm_xdl_f16_f16_f16/common.hpp`  (+43/-0)
- `library/src/tensor_operation_instance/gpu/gemm/device_gemm_xdl_f16_f16_f16/km_kn_mn_default_pipeline_v2_instance.cpp`  (+42/-0)
- `library/src/tensor_operation_instance/gpu/gemm/device_gemm_xdl_f16_f16_f16/km_kn_mn_interwave_pipeline_v1_instance.cpp`  (+42/-0)
- `library/src/tensor_operation_instance/gpu/gemm/device_gemm_xdl_f16_f16_f16/km_nk_mn_default_pipeline_v2_instance.cpp`  (+42/-0)
- `library/src/tensor_operation_instance/gpu/gemm/device_gemm_xdl_f16_f16_f16/km_nk_mn_interwave_pipeline_v1_instance.cpp`  (+42/-0)

## Key added lines (kernel files)

**`include/ck/tensor_operation/gpu/grid/gridwise_gemm_pipeline_v2.hpp`**
```
__builtin_amdgcn_iglp_opt(CK_EXPERIMENTAL_PIPELINE_V2_IGLP_OPT);
```

**`include/ck/tensor_operation/gpu/grid/gridwise_gemm_xdlops_v2r3.hpp`**
```
__attribute__((amdgpu_waves_per_eu(CK_MIN_WAVES_PER_EU, CK_MAX_WAVES_PER_EU)))
__attribute__((amdgpu_waves_per_eu(CK_MIN_WAVES_PER_EU, CK_MAX_WAVES_PER_EU)))
```

**`library/src/tensor_operation_instance/gpu/gemm/device_gemm_xdl_f16_f16_f16/common.hpp`**
```
namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {
```

**`library/src/tensor_operation_instance/gpu/gemm/device_gemm_xdl_f16_f16_f16/km_kn_mn_add_instance.cpp`**
```
namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {
```

**`library/src/tensor_operation_instance/gpu/gemm/device_gemm_xdl_f16_f16_f16/km_kn_mn_default_pipeline_v1_instance.cpp`**
```
namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {
```
