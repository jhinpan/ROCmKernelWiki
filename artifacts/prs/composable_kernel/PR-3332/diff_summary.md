# Diff summary

- **files changed:** 13
- **lines:** +1345 / -78
- **kernel-ish files:** 11

## Files (by churn)

- `include/ck/tensor_operation/gpu/device/impl/device_batched_gemm_reduce_wmma_cshuffle_v3.hpp`  (+799/-0)
- `test/batched_gemm_reduce/batched_gemm_reduce_fp16.cpp`  (+119/-0)
- `library/src/tensor_operation_instance/gpu/batched_gemm_reduce/device_batched_gemm_reduce_wmma_cshuffle_v3_f16_f16_f16_f32_f32_gkm_gkn_gmn_instance.cpp`  (+88/-0)
- `library/src/tensor_operation_instance/gpu/batched_gemm_reduce/device_batched_gemm_reduce_wmma_cshuffle_v3_f16_f16_f16_f32_f32_gkm_gnk_gmn_instance.cpp`  (+88/-0)
- `library/src/tensor_operation_instance/gpu/batched_gemm_reduce/device_batched_gemm_reduce_wmma_cshuffle_v3_f16_f16_f16_f32_f32_gmk_gkn_gmn_instance.cpp`  (+87/-0)
- `library/src/tensor_operation_instance/gpu/batched_gemm_reduce/device_batched_gemm_reduce_wmma_cshuffle_v3_f16_f16_f16_f32_f32_gmk_gnk_gmn_instance.cpp`  (+86/-0)
- `profiler/include/profiler/profile_batched_gemm_reduce_impl.hpp`  (+64/-3)
- `test/batched_gemm_reduce/batched_gemm_reduce_fp16_xdl.cpp`  (+0/-67)
- `test/batched_gemm_reduce/CMakeLists.txt`  (+6/-4)
- `library/src/tensor_operation_instance/gpu/batched_gemm_reduce/CMakeLists.txt`  (+5/-1)
- `library/src/tensor_operation_instance/gpu/batched_gemm_reduce/device_batched_gemm_reduce_xdl_cshuffle_f16_f16_f16_f32_f32_gkm_gkn_gmn_instance.cpp`  (+1/-1)
- `library/src/tensor_operation_instance/gpu/batched_gemm_reduce/device_batched_gemm_reduce_xdl_cshuffle_f16_f16_f16_f32_f32_gkm_gnk_gmn_instance.cpp`  (+1/-1)
- `library/src/tensor_operation_instance/gpu/batched_gemm_reduce/device_batched_gemm_reduce_xdl_cshuffle_f16_f16_f16_f32_f32_gmk_gkn_gmn_instance.cpp`  (+1/-1)

## Key added lines (kernel files)

**`include/ck/tensor_operation/gpu/device/impl/device_batched_gemm_reduce_wmma_cshuffle_v3.hpp`**
```
namespace ck {
template <typename GridwiseGemm,
typename ReduceTrait,
typename ComputePtrOffsetOfStridedBatch,
```

**`library/src/tensor_operation_instance/gpu/batched_gemm_reduce/device_batched_gemm_reduce_wmma_cshuffle_v3_f16_f16_f16_f32_f32_gkm_gkn_gmn_instance.cpp`**
```
namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {
```

**`library/src/tensor_operation_instance/gpu/batched_gemm_reduce/device_batched_gemm_reduce_wmma_cshuffle_v3_f16_f16_f16_f32_f32_gkm_gnk_gmn_instance.cpp`**
```
namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {
```

**`library/src/tensor_operation_instance/gpu/batched_gemm_reduce/device_batched_gemm_reduce_wmma_cshuffle_v3_f16_f16_f16_f32_f32_gmk_gkn_gmn_instance.cpp`**
```
namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {
```

**`library/src/tensor_operation_instance/gpu/batched_gemm_reduce/device_batched_gemm_reduce_wmma_cshuffle_v3_f16_f16_f16_f32_f32_gmk_gnk_gmn_instance.cpp`**
```
namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {
```
