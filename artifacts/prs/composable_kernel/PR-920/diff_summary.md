# Diff summary

- **files changed:** 8
- **lines:** +350 / -1
- **kernel-ish files:** 7

## Files (by churn)

- `library/src/tensor_operation_instance/gpu/gemm/device_gemm_xdl_c_shuffle_fp8_fp8_fp8_km_kn_mn_instance.cpp`  (+69/-0)
- `library/src/tensor_operation_instance/gpu/gemm/device_gemm_xdl_c_shuffle_fp8_fp8_fp8_km_nk_mn_instance.cpp`  (+69/-0)
- `library/src/tensor_operation_instance/gpu/gemm/device_gemm_xdl_c_shuffle_fp8_fp8_fp8_mk_kn_mn_instance.cpp`  (+69/-0)
- `library/src/tensor_operation_instance/gpu/gemm/device_gemm_xdl_c_shuffle_fp8_fp8_fp8_mk_nk_mn_instance.cpp`  (+66/-0)
- `library/include/ck/library/tensor_operation_instance/gpu/gemm.hpp`  (+43/-0)
- `profiler/src/profile_gemm.cpp`  (+23/-1)
- `profiler/include/profiler/profile_gemm_impl.hpp`  (+6/-0)
- `library/src/tensor_operation_instance/gpu/gemm/CMakeLists.txt`  (+5/-0)

## Key added lines (kernel files)

**`library/include/ck/library/tensor_operation_instance/gpu/gemm.hpp`**
```
void add_device_gemm_xdl_c_shuffle_f8_f8_f8_km_kn_mn_instances(
std::vector<std::unique_ptr<
DeviceGemm<Col, Row, Row, F8, F8, F8, PassThrough, PassThrough, PassThrough>>>& instances);
void add_device_gemm_xdl_c_shuffle_f8_f8_f8_km_nk_mn_instances(
```

**`library/src/tensor_operation_instance/gpu/gemm/device_gemm_xdl_c_shuffle_fp8_fp8_fp8_km_kn_mn_instance.cpp`**
```
namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {
```

**`library/src/tensor_operation_instance/gpu/gemm/device_gemm_xdl_c_shuffle_fp8_fp8_fp8_km_nk_mn_instance.cpp`**
```
namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {
```

**`library/src/tensor_operation_instance/gpu/gemm/device_gemm_xdl_c_shuffle_fp8_fp8_fp8_mk_kn_mn_instance.cpp`**
```
namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {
```

**`library/src/tensor_operation_instance/gpu/gemm/device_gemm_xdl_c_shuffle_fp8_fp8_fp8_mk_nk_mn_instance.cpp`**
```
namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {
```
