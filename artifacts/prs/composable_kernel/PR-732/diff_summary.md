# Diff summary

- **files changed:** 15 (diff was byte-capped; summary is partial)
- **lines:** +2196 / -0
- **kernel-ish files:** 14

## Files (by churn)

- `include/ck/tensor_operation/gpu/device/impl/device_batched_gemm_multiple_d_dl.hpp`  (+796/-0)
- `library/include/ck/library/tensor_operation_instance/gpu/batched_gemm_multi_d.hpp`  (+337/-0)
- `library/src/tensor_operation_instance/gpu/batched_gemm_multi_d/device_batched_gemm_multi_d_dl_f16_f16_f16_gkm_gkn_gmn_instance.cpp`  (+95/-0)
- `library/src/tensor_operation_instance/gpu/batched_gemm_multi_d/device_batched_gemm_multi_d_dl_f16_f16_f16_gkm_gnk_gmn_instance.cpp`  (+95/-0)
- `library/src/tensor_operation_instance/gpu/batched_gemm_multi_d/device_batched_gemm_multi_d_dl_f16_f16_f16_gmk_gkn_gmn_instance.cpp`  (+95/-0)
- `library/src/tensor_operation_instance/gpu/batched_gemm_multi_d/device_batched_gemm_multi_d_dl_f16_f16_f16_gmk_gnk_gmn_instance.cpp`  (+95/-0)
- `library/src/tensor_operation_instance/gpu/batched_gemm_multi_d/device_batched_gemm_multi_d_dl_i8_i8_i8_gkm_gkn_gmn_instance.cpp`  (+93/-0)
- `library/src/tensor_operation_instance/gpu/batched_gemm_multi_d/device_batched_gemm_multi_d_dl_i8_i8_i8_gkm_gnk_gmn_instance.cpp`  (+93/-0)
- `library/src/tensor_operation_instance/gpu/batched_gemm_multi_d/device_batched_gemm_multi_d_dl_i8_i8_i8_gkm_gkn_gmn_irregular_instance.cpp`  (+90/-0)
- `library/src/tensor_operation_instance/gpu/batched_gemm_multi_d/device_batched_gemm_multi_d_dl_f16_f16_f16_gkm_gkn_gmn_irregular_instance.cpp`  (+84/-0)
- `library/src/tensor_operation_instance/gpu/batched_gemm_multi_d/device_batched_gemm_multi_d_dl_f16_f16_f16_gkm_gnk_gmn_irregular_instance.cpp`  (+83/-0)
- `library/src/tensor_operation_instance/gpu/batched_gemm_multi_d/device_batched_gemm_multi_d_dl_f16_f16_f16_gmk_gkn_gmn_irregular_instance.cpp`  (+83/-0)
- `library/src/tensor_operation_instance/gpu/batched_gemm_multi_d/device_batched_gemm_multi_d_dl_f16_f16_f16_gmk_gnk_gmn_irregular_instance.cpp`  (+83/-0)
- `library/src/tensor_operation_instance/gpu/batched_gemm_multi_d/device_batched_gemm_multi_d_dl_i8_i8_i8_gkm_gnk_gmn_irregular_instance.cpp`  (+56/-0)
- `library/src/tensor_operation_instance/gpu/batched_gemm_multi_d/CMakeLists.txt`  (+18/-0)

## Key added lines (kernel files)

**`include/ck/tensor_operation/gpu/device/impl/device_batched_gemm_multiple_d_dl.hpp`**
```
namespace ck {
namespace tensor_operation {
namespace device {
template <typename GridwiseGemm,
```

**`library/include/ck/library/tensor_operation_instance/gpu/batched_gemm_multi_d.hpp`**
```
namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {
```

**`library/src/tensor_operation_instance/gpu/batched_gemm_multi_d/device_batched_gemm_multi_d_dl_f16_f16_f16_gkm_gkn_gmn_instance.cpp`**
```
namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {
```

**`library/src/tensor_operation_instance/gpu/batched_gemm_multi_d/device_batched_gemm_multi_d_dl_f16_f16_f16_gkm_gkn_gmn_irregular_instance.cpp`**
```
namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {
```

**`library/src/tensor_operation_instance/gpu/batched_gemm_multi_d/device_batched_gemm_multi_d_dl_f16_f16_f16_gkm_gnk_gmn_instance.cpp`**
```
namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {
```
