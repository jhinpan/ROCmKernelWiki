# Diff summary

- **files changed:** 13
- **lines:** +2490 / -16
- **kernel-ish files:** 11

## Files (by churn)

- `include/ck/tensor_operation/gpu/device/impl/device_grouped_gemm_multiple_d_splitk_xdl_cshuffle_two_stage.hpp`  (+987/-0)
- `example/15_grouped_gemm/grouped_gemm_multiple_d_splitk_xdl_fp16.cpp`  (+394/-0)
- `profiler/include/profiler/profile_grouped_gemm_two_stage_impl.hpp`  (+366/-0)
- `library/include/ck/library/reference_tensor_operation/cpu/reference_gemm_multiple_d.hpp`  (+175/-0)
- `profiler/src/profile_grouped_gemm_two_stage.cpp`  (+157/-0)
- `include/ck/tensor_operation/gpu/device/device_grouped_gemm_multiple_d_splitk.hpp`  (+136/-0)
- `library/src/tensor_operation_instance/gpu/grouped_gemm/device_grouped_gemm_multiple_d_splitk_xdl_two_stage_bf16_i8_bf16_mk_kn_mn_instance.cpp`  (+99/-0)
- `library/src/tensor_operation_instance/gpu/grouped_gemm/device_grouped_gemm_multiple_d_splitk_xdl_two_stage_f16_f16_f16_mk_kn_mn_instance.cpp`  (+96/-0)
- `library/include/ck/library/tensor_operation_instance/gpu/grouped_gemm.hpp`  (+47/-3)
- `include/ck/tensor_operation/gpu/device/impl/device_grouped_gemm_xdl_splitk_cshuffle.hpp`  (+21/-6)
- `include/ck/tensor_operation/gpu/device/impl/device_elementwise_impl.hpp`  (+9/-7)
- `library/src/tensor_operation_instance/gpu/grouped_gemm/CMakeLists.txt`  (+2/-0)
- `profiler/src/CMakeLists.txt`  (+1/-0)

## Key added lines (kernel files)

**`example/15_grouped_gemm/grouped_gemm_multiple_d_splitk_xdl_fp16.cpp`**
```
template <ck::index_t... Is>
using S = ck::Sequence<Is...>;
using F16 = ck::half_t;
using F32 = float;
```

**`include/ck/tensor_operation/gpu/device/device_grouped_gemm_multiple_d_splitk.hpp`**
```
namespace ck {
namespace tensor_operation {
namespace device {
template <index_t NumDTensor = 0>
```

**`include/ck/tensor_operation/gpu/device/impl/device_elementwise_impl.hpp`**
```
index_t NumDim,                 // The max dim of input tensors
index_t MPerThread,             // How many elements per thread to read
typename InScalarPerVectorSeq,  // Scalar per vec for each Input
typename OutScalarPerVectorSeq> // Scalar per vec for each Output
```

**`include/ck/tensor_operation/gpu/device/impl/device_grouped_gemm_multiple_d_splitk_xdl_cshuffle_two_stage.hpp`**
```
namespace ck {
namespace tensor_operation {
namespace device {
template <typename ALayout,
```

**`include/ck/tensor_operation/gpu/device/impl/device_grouped_gemm_xdl_splitk_cshuffle.hpp`**
```
InMemoryDataOperationEnum CGlobalMemoryDataOperation,
typename AElementwiseOperation = ck::tensor_operation::element_wise::PassThrough,
typename BElementwiseOperation = ck::tensor_operation::element_wise::PassThrough,
typename CElementwiseOperation = ck::tensor_operation::element_wise::PassThrough>
```
