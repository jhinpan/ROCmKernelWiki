# Diff summary

- **files changed:** 24
- **lines:** +3517 / -120
- **kernel-ish files:** 21

## Files (by churn)

- `include/ck/tensor_operation/gpu/device/impl/device_grouped_gemm_multi_abd_wmma_fixed_nk.hpp`  (+899/-0)
- `profiler/include/profiler/profile_grouped_gemm_multi_abd_fixed_nk_impl.hpp`  (+534/-0)
- `example/59_grouped_gemm_multi_ABD/grouped_gemm_multi_abd_wmma_fixed_nk_bias_bf16_i8.cpp`  (+400/-0)
- `example/59_grouped_gemm_multi_ABD/grouped_gemm_multi_abd_wmma_fixed_nk_bias_fp16.cpp`  (+396/-0)
- `library/include/ck/library/tensor_operation_instance/gpu/grouped_gemm_multi_abd_fixed_nk.hpp`  (+294/-1)
- `test/grouped_gemm/test_grouped_gemm_multi_abd_fixed_nk.cpp`  (+256/-0)
- `library/include/ck/library/reference_tensor_operation/cpu/reference_gemm_multi_abd.hpp`  (+194/-0)
- `library/src/tensor_operation_instance/gpu/grouped_gemm_fixed_nk_multi_abd/device_grouped_gemm_wmma_fixed_nk_bias_gelu_bf16_i8_bf16_km_kn_mn_instance.cpp`  (+144/-0)
- `library/src/tensor_operation_instance/gpu/grouped_gemm_fixed_nk_multi_abd/device_grouped_gemm_wmma_fixed_nk_bias_gelu_bf16_i8_bf16_mk_kn_mn_instance.cpp`  (+144/-0)
- `library/src/tensor_operation_instance/gpu/grouped_gemm_fixed_nk_multi_abd/device_grouped_gemm_wmma_fixed_nk_bias_gelu_bf16_i8_bf16_mk_nk_mn_instance.cpp`  (+144/-0)
- `profiler/include/profiler/profile_gemm_multi_abd_impl.hpp`  (+16/-72)
- `example/59_grouped_gemm_multi_ABD/grouped_gemm_multi_abd_xdl_fixed_nk_bias_fp16.cpp`  (+14/-19)
- `example/59_grouped_gemm_multi_ABD/grouped_gemm_multi_abd_xdl_fixed_nk_bias_bf16_i8.cpp`  (+10/-17)
- `include/ck/tensor_operation/gpu/device/impl/device_grouped_gemm_multi_abd_xdl_fixed_nk.hpp`  (+26/-1)
- `library/src/tensor_operation_instance/gpu/grouped_gemm_fixed_nk_multi_abd/device_grouped_gemm_xdl_fixed_nk_bf16_i8_bf16_km_kn_mn_common.hpp`  (+6/-4)

## Key added lines (kernel files)

**`example/59_grouped_gemm_multi_ABD/grouped_gemm_multi_abd_wmma_fixed_nk_bias_bf16_i8.cpp`**
```
using ::ck::DeviceMem;
using ::ck::hip_check_error;
using ::ck::HostTensorDescriptor;
using ::ck::Tensor;
```

**`example/59_grouped_gemm_multi_ABD/grouped_gemm_multi_abd_wmma_fixed_nk_bias_fp16.cpp`**
```
using ::ck::DeviceMem;
using ::ck::hip_check_error;
using ::ck::HostTensorDescriptor;
using ::ck::Tensor;
```

**`example/59_grouped_gemm_multi_ABD/grouped_gemm_multi_abd_xdl_fixed_nk_bias_bf16_i8.cpp`**
```
a0_tensors_device.emplace_back(std::make_unique<DeviceMem>(
sizeof(A0DataType) * problem_size.Ms[i] * problem_size.Ks[i]));
c_tensors_device.emplace_back(std::make_unique<DeviceMem>(
sizeof(EDataType) * problem_size.Ms[i] * problem_size.Ns[i]));
```

**`example/59_grouped_gemm_multi_ABD/grouped_gemm_multi_abd_xdl_fixed_nk_bias_fp16.cpp`**
```
using D0DataType       = F16;
using EDataType        = F16;
a0_tensors_device.emplace_back(std::make_unique<DeviceMem>(
sizeof(A0DataType) * problem_size.Ms[i] * problem_size.Ks[i]));
```

**`include/ck/tensor_operation/gpu/device/impl/device_grouped_gemm_multi_abd_wmma_fixed_nk.hpp`**
```
namespace ck {
namespace tensor_operation {
namespace device {
template <typename GridwiseGemm,
```
