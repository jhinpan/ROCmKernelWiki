# Diff summary

- **files changed:** 12
- **lines:** +948 / -7
- **kernel-ish files:** 8

## Files (by churn)

- `test/gemm_mx/test_gemm_mx_util.hpp`  (+498/-0)
- `library/include/ck/library/tensor_operation_instance/gpu/gemm_mx.hpp`  (+111/-0)
- `test/gemm_mx/test_gemm_mx.cpp`  (+108/-0)
- `library/src/tensor_operation_instance/gpu/gemm_mx/device_gemm_mx_xdl_f8_f8_bf16/device_gemm_mx_xdl_f8_f8_bf16_mk_nk_mn.hpp`  (+63/-0)
- `library/src/tensor_operation_instance/gpu/gemm_mx/device_gemm_mx_xdl_f8_f8_f16/device_gemm_mx_xdl_f8_f8_f16_mk_nk_mn.hpp`  (+63/-0)
- `library/src/tensor_operation_instance/gpu/gemm_mx/device_gemm_mx_xdl_f8_f8_bf16/device_gemm_mx_xdl_f8_f8_bf16_mk_nk_mn_default_instance.cpp`  (+32/-0)
- `library/src/tensor_operation_instance/gpu/gemm_mx/device_gemm_mx_xdl_f8_f8_f16/device_gemm_mx_xdl_f8_f8_f16_mk_nk_mn_default_instance.cpp`  (+32/-0)
- `include/ck/tensor_operation/gpu/device/impl/device_gemm_xdl_cshuffle_v3_mx.hpp`  (+9/-7)
- `library/src/tensor_operation_instance/gpu/gemm_mx/CMakeLists.txt`  (+14/-0)
- `library/src/tensor_operation_instance/gpu/CMakeLists.txt`  (+13/-0)
- `test/gemm_mx/CMakeLists.txt`  (+4/-0)
- `test/CMakeLists.txt`  (+1/-0)

## Key added lines (kernel files)

**`include/ck/tensor_operation/gpu/device/impl/device_gemm_xdl_cshuffle_v3_mx.hpp`**
```
typename GemmAccDataType, // TODO: always float
```

**`library/include/ck/library/tensor_operation_instance/gpu/gemm_mx.hpp`**
```
namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {
```

**`library/src/tensor_operation_instance/gpu/gemm_mx/device_gemm_mx_xdl_f8_f8_bf16/device_gemm_mx_xdl_f8_f8_bf16_mk_nk_mn.hpp`**
```
namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {
```

**`library/src/tensor_operation_instance/gpu/gemm_mx/device_gemm_mx_xdl_f8_f8_bf16/device_gemm_mx_xdl_f8_f8_bf16_mk_nk_mn_default_instance.cpp`**
```
namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {
```

**`library/src/tensor_operation_instance/gpu/gemm_mx/device_gemm_mx_xdl_f8_f8_f16/device_gemm_mx_xdl_f8_f8_f16_mk_nk_mn.hpp`**
```
namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {
```
