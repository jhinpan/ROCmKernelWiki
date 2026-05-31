# Diff summary

- **files changed:** 11
- **lines:** +265 / -9
- **kernel-ish files:** 9

## Files (by churn)

- `library/src/tensor_operation_instance/gpu/gemm_mx/device_gemm_mx_xdl_f6_f6_f16/device_gemm_mx_xdl_f6_f6_f16_mk_nk_mn.hpp`  (+67/-0)
- `library/src/tensor_operation_instance/gpu/gemm_mx/device_gemm_mx_xdl_bf6_bf6_bf16/device_gemm_mx_xdl_bf6_bf6_bf16_mk_nk_mn.hpp`  (+66/-0)
- `library/include/ck/library/tensor_operation_instance/gpu/gemm_mx.hpp`  (+40/-0)
- `library/src/tensor_operation_instance/gpu/gemm_mx/device_gemm_mx_xdl_bf6_bf6_bf16/device_gemm_mx_xdl_bf6_bf6_bf16_mk_nk_mn_default_instance.cpp`  (+32/-0)
- `library/src/tensor_operation_instance/gpu/gemm_mx/device_gemm_mx_xdl_f6_f6_f16/device_gemm_mx_xdl_f6_f6_f16_mk_nk_mn_default_instance.cpp`  (+32/-0)
- `profiler/include/profiler/profile_gemm_mx_impl.hpp`  (+15/-5)
- `test/gemm_mx/test_gemm_mx.cpp`  (+5/-3)
- `library/src/tensor_operation_instance/gpu/gemm_mx/CMakeLists.txt`  (+4/-0)
- `library/include/ck/library/tensor_operation_instance/device_operation_instance_factory.hpp`  (+2/-0)
- `test/gemm_mx/test_gemm_mx_util.hpp`  (+1/-1)
- `test/gemm_mx/CMakeLists.txt`  (+1/-0)

## Key added lines (kernel files)

**`library/include/ck/library/tensor_operation_instance/device_operation_instance_factory.hpp`**
```
using F6   = ck::f6x16_pk_t;
using BF6  = ck::bf6x16_pk_t;
```

**`library/include/ck/library/tensor_operation_instance/gpu/gemm_mx.hpp`**
```
void add_device_gemm_mx_xdl_f6_f6_f16_mk_nk_mn_default_instances(
std::vector<std::unique_ptr<DeviceGemmMX<Row,
PassThrough,
PassThrough,
```

**`library/src/tensor_operation_instance/gpu/gemm_mx/device_gemm_mx_xdl_bf6_bf6_bf16/device_gemm_mx_xdl_bf6_bf6_bf16_mk_nk_mn.hpp`**
```
namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {
```

**`library/src/tensor_operation_instance/gpu/gemm_mx/device_gemm_mx_xdl_bf6_bf6_bf16/device_gemm_mx_xdl_bf6_bf6_bf16_mk_nk_mn_default_instance.cpp`**
```
namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {
```

**`library/src/tensor_operation_instance/gpu/gemm_mx/device_gemm_mx_xdl_f6_f6_f16/device_gemm_mx_xdl_f6_f6_f16_mk_nk_mn.hpp`**
```
namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {
```
