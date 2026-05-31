# Diff summary

- **files changed:** 51
- **lines:** +2101 / -10
- **kernel-ish files:** 47

## Files (by churn)

- `library/include/ck/library/tensor_operation_instance/gpu/gemm_universal_streamk.hpp`  (+500/-0)
- `library/src/tensor_operation_instance/gpu/gemm_universal_streamk/device_gemm_xdl_universal_streamk_bf16_bf16_bf16/device_gemm_xdl_universal_streamk_bf16_bf16_bf16_km_nk_mn.hpp`  (+97/-0)
- `library/src/tensor_operation_instance/gpu/gemm_universal_streamk/device_gemm_xdl_universal_streamk_bf16_bf16_bf16/device_gemm_xdl_universal_streamk_bf16_bf16_bf16_mk_nk_mn.hpp`  (+93/-0)
- `library/src/tensor_operation_instance/gpu/gemm_universal_streamk/device_gemm_xdl_universal_streamk_bf16_bf16_bf16/device_gemm_xdl_universal_streamk_bf16_bf16_bf16_km_kn_mn.hpp`  (+91/-0)
- `library/src/tensor_operation_instance/gpu/gemm_universal_streamk/device_gemm_xdl_universal_streamk_bf16_bf16_bf16/device_gemm_xdl_universal_streamk_bf16_bf16_bf16_mk_kn_mn.hpp`  (+89/-0)
- `example/01_gemm/gemm_xdl_bf16_streamk_v3.cpp`  (+59/-0)
- `library/src/tensor_operation_instance/gpu/gemm_universal_streamk/CMakeLists.txt`  (+38/-1)
- `library/src/tensor_operation_instance/gpu/gemm_universal_streamk/device_gemm_xdl_universal_streamk_bf16_bf16_bf16/device_gemm_xdl_universal_streamk_bf16_bf16_bf16_km_kn_mn_mem_v1_default_instance.cpp`  (+31/-0)
- `library/src/tensor_operation_instance/gpu/gemm_universal_streamk/device_gemm_xdl_universal_streamk_bf16_bf16_bf16/device_gemm_xdl_universal_streamk_bf16_bf16_bf16_km_kn_mn_mem_v1_kpadding_instance.cpp`  (+31/-0)
- `library/src/tensor_operation_instance/gpu/gemm_universal_streamk/device_gemm_xdl_universal_streamk_bf16_bf16_bf16/device_gemm_xdl_universal_streamk_bf16_bf16_bf16_km_kn_mn_mem_v1_mnkpadding_instance.cpp`  (+31/-0)
- `library/src/tensor_operation_instance/gpu/gemm_universal_streamk/device_gemm_xdl_universal_streamk_bf16_bf16_bf16/device_gemm_xdl_universal_streamk_bf16_bf16_bf16_km_kn_mn_mem_v2_default_instance.cpp`  (+31/-0)
- `library/src/tensor_operation_instance/gpu/gemm_universal_streamk/device_gemm_xdl_universal_streamk_bf16_bf16_bf16/device_gemm_xdl_universal_streamk_bf16_bf16_bf16_km_kn_mn_mem_v2_kpadding_instance.cpp`  (+31/-0)
- `library/src/tensor_operation_instance/gpu/gemm_universal_streamk/device_gemm_xdl_universal_streamk_bf16_bf16_bf16/device_gemm_xdl_universal_streamk_bf16_bf16_bf16_km_kn_mn_mem_v2_mnkpadding_instance.cpp`  (+31/-0)
- `library/src/tensor_operation_instance/gpu/gemm_universal_streamk/device_gemm_xdl_universal_streamk_bf16_bf16_bf16/device_gemm_xdl_universal_streamk_bf16_bf16_bf16_km_nk_mn_mem_v1_default_instance.cpp`  (+31/-0)
- `library/src/tensor_operation_instance/gpu/gemm_universal_streamk/device_gemm_xdl_universal_streamk_bf16_bf16_bf16/device_gemm_xdl_universal_streamk_bf16_bf16_bf16_km_nk_mn_mem_v1_kpadding_instance.cpp`  (+31/-0)

## Key added lines (kernel files)

**`example/01_gemm/gemm_xdl_bf16_streamk_v3.cpp`**
```
using ADataType        = ck::bhalf_t;
using BDataType        = ck::bhalf_t;
using CDataType        = ck::bhalf_t;
using AccDataType      = float;
```

**`include/ck/tensor_operation/gpu/device/impl/device_gemm_xdl_cshuffle_streamk_v3.hpp`**
```
if(!is_bf16_atomic_supported() && std::is_same_v<CDataType, ck::bhalf_t> &&
arg.Streamk_sel > 0)
return false;
```

**`library/include/ck/library/tensor_operation_instance/gpu/gemm_universal_streamk.hpp`**
```
void add_device_gemm_xdl_universal_streamk_bf16_bf16_bf16_mk_kn_mn_comp_default_instances(
std::vector<std::unique_ptr<DeviceGemm_Streamk_V2<Row,
PassThrough,
PassThrough,
```

**`library/src/tensor_operation_instance/gpu/gemm_universal_streamk/device_gemm_xdl_universal_streamk_bf16_bf16_bf16/device_gemm_xdl_universal_streamk_bf16_bf16_bf16_km_kn_mn.hpp`**
```
namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {
```

**`library/src/tensor_operation_instance/gpu/gemm_universal_streamk/device_gemm_xdl_universal_streamk_bf16_bf16_bf16/device_gemm_xdl_universal_streamk_bf16_bf16_bf16_km_kn_mn_comp_default_instance.cpp`**
```
namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {
```
