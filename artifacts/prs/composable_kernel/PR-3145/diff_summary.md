# Diff summary

- **files changed:** 12
- **lines:** +1568 / -12
- **kernel-ish files:** 10

## Files (by churn)

- `include/ck/tensor_operation/gpu/device/impl/device_gemm_reduce_wmma_cshuffle_v3.hpp`  (+661/-0)
- `include/ck/tensor_operation/gpu/grid/epilogue_cshuffle_v3_reduce_wmma.hpp`  (+470/-0)
- `library/src/tensor_operation_instance/gpu/gemm_reduce/device_gemm_reduce_wmma_cshuffle_v3_f16_f16_f16_f32_f32_km_kn_mn_instance.cpp`  (+88/-0)
- `library/src/tensor_operation_instance/gpu/gemm_reduce/device_gemm_reduce_wmma_cshuffle_v3_f16_f16_f16_f32_f32_km_nk_mn_instance.cpp`  (+88/-0)
- `library/src/tensor_operation_instance/gpu/gemm_reduce/device_gemm_reduce_wmma_cshuffle_v3_f16_f16_f16_f32_f32_mk_kn_mn_instance.cpp`  (+88/-0)
- `library/src/tensor_operation_instance/gpu/gemm_reduce/device_gemm_reduce_wmma_cshuffle_v3_f16_f16_f16_f32_f32_mk_nk_mn_instance.cpp`  (+86/-0)
- `profiler/include/profiler/profile_gemm_reduce_impl.hpp`  (+49/-6)
- `include/ck/tensor_operation/gpu/grid/gridwise_gemm_wmma_cshuffle_v3_common.hpp`  (+24/-0)
- `test/gemm_reduce/CMakeLists.txt`  (+6/-4)
- `library/src/tensor_operation_instance/gpu/gemm_reduce/CMakeLists.txt`  (+6/-1)
- `test/gemm_reduce/gemm_reduce_fp16.cpp`  (+1/-1)
- `include/ck/tensor_operation/gpu/grid/epilogue_cshuffle_v3_wmma_base.hpp`  (+1/-0)

## Key added lines (kernel files)

**`include/ck/tensor_operation/gpu/device/impl/device_gemm_reduce_wmma_cshuffle_v3.hpp`**
```
namespace ck {
template <typename GridwiseGemm,
typename ReduceTrait,
bool HasMainKBlockLoop,
```

**`include/ck/tensor_operation/gpu/grid/epilogue_cshuffle_v3_reduce_wmma.hpp`**
```
namespace ck {
template <typename ReduceAccDataType,
typename ReducePtrsGlobal,
typename ReduceOperations,
```

**`include/ck/tensor_operation/gpu/grid/gridwise_gemm_wmma_cshuffle_v3_common.hpp`**
```
template <typename ReduceTrait>
using EpilogueReduceCShuffle = EpilogueReduceCShuffle<
DsDataType,
EDataType,
```

**`library/src/tensor_operation_instance/gpu/gemm_reduce/device_gemm_reduce_wmma_cshuffle_v3_f16_f16_f16_f32_f32_km_kn_mn_instance.cpp`**
```
namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {
```

**`library/src/tensor_operation_instance/gpu/gemm_reduce/device_gemm_reduce_wmma_cshuffle_v3_f16_f16_f16_f32_f32_km_nk_mn_instance.cpp`**
```
namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {
```
