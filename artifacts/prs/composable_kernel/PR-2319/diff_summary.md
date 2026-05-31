# Diff summary

- **files changed:** 14
- **lines:** +1684 / -22
- **kernel-ish files:** 11

## Files (by churn)

- `include/ck/tensor_operation/gpu/device/impl/device_batched_gemm_wmma_cshuffle_v3.hpp`  (+759/-0)
- `test/batched_gemm/test_batched_gemm_wmma.cpp`  (+193/-0)
- `library/include/ck/library/tensor_operation_instance/gpu/batched_gemm.hpp`  (+104/-1)
- `library/src/tensor_operation_instance/gpu/batched_gemm/device_batched_gemm_wmma_universal_bf16_bf16_bf16_gmk_gnk_gmn_instance.cpp`  (+79/-0)
- `library/src/tensor_operation_instance/gpu/batched_gemm/device_batched_gemm_wmma_universal_f16_f16_f16_gmk_gnk_gmn_instance.cpp`  (+78/-0)
- `library/src/tensor_operation_instance/gpu/batched_gemm/device_batched_gemm_wmma_universal_bf16_bf16_bf16_gmk_gkn_gmn_instance.cpp`  (+76/-0)
- `library/src/tensor_operation_instance/gpu/batched_gemm/device_batched_gemm_wmma_universal_f16_f16_f16_gmk_gkn_gmn_instance.cpp`  (+75/-0)
- `library/src/tensor_operation_instance/gpu/batched_gemm/device_batched_gemm_wmma_universal_bf16_bf16_bf16_gkm_gnk_gmn_instance.cpp`  (+73/-0)
- `library/src/tensor_operation_instance/gpu/batched_gemm/device_batched_gemm_wmma_universal_f16_f16_f16_gkm_gnk_gmn_instance.cpp`  (+72/-0)
- `library/src/tensor_operation_instance/gpu/batched_gemm/device_batched_gemm_wmma_universal_bf16_bf16_bf16_gkm_gkn_gmn_instance.cpp`  (+71/-0)
- `library/src/tensor_operation_instance/gpu/batched_gemm/device_batched_gemm_wmma_universal_f16_f16_f16_gkm_gkn_gmn_instance.cpp`  (+70/-0)
- `library/src/tensor_operation_instance/gpu/batched_gemm/CMakeLists.txt`  (+25/-17)
- `test/batched_gemm/CMakeLists.txt`  (+7/-2)
- `profiler/src/CMakeLists.txt`  (+2/-2)

## Key added lines (kernel files)

**`include/ck/tensor_operation/gpu/device/impl/device_batched_gemm_wmma_cshuffle_v3.hpp`**
```
namespace ck {
namespace tensor_operation {
namespace device {
template <typename GridwiseGemm,
```

**`library/include/ck/library/tensor_operation_instance/gpu/batched_gemm.hpp`**
```
void add_device_batched_gemm_wmma_universal_f16_f16_f16_gkm_gkn_gmn_instances(
std::vector<std::unique_ptr<
DeviceBatchedGemm<Col, Row, Row, F16, F16, F16, PassThrough, PassThrough, PassThrough>>>&
instances);
```

**`library/src/tensor_operation_instance/gpu/batched_gemm/device_batched_gemm_wmma_universal_bf16_bf16_bf16_gkm_gkn_gmn_instance.cpp`**
```
namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {
```

**`library/src/tensor_operation_instance/gpu/batched_gemm/device_batched_gemm_wmma_universal_bf16_bf16_bf16_gkm_gnk_gmn_instance.cpp`**
```
namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {
```

**`library/src/tensor_operation_instance/gpu/batched_gemm/device_batched_gemm_wmma_universal_bf16_bf16_bf16_gmk_gkn_gmn_instance.cpp`**
```
namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {
```
