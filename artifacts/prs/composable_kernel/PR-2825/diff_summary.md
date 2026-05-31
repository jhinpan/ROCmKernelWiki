# Diff summary

- **files changed:** 22
- **lines:** +1355 / -100
- **kernel-ish files:** 18

## Files (by churn)

- `include/ck/tensor_operation/gpu/device/impl/device_batched_gemm_wmma_cshuffle_v3_b_scale.hpp`  (+836/-0)
- `profiler/include/profiler/profile_batched_gemm_b_scale_impl.hpp`  (+60/-51)
- `test/batched_gemm_b_scale/test_batched_gemm_b_scale_util.hpp`  (+108/-0)
- `library/include/ck/library/reference_tensor_operation/cpu/reference_batched_gemm.hpp`  (+58/-18)
- `library/src/tensor_operation_instance/gpu/batched_gemm_b_scale/device_batched_gemm_b_scale_wmma_f16_i4_f16/device_batched_gemm_b_scale_wmma_f16_i4_f16_mk_nk_mn.hpp`  (+72/-0)
- `test/batched_gemm_b_scale/test_batched_gemm_b_scale_ut_cases.inc`  (+49/-0)
- `test/batched_gemm_b_scale/test_batched_gemm_b_scale_wmma.cpp`  (+45/-0)
- `profiler/include/profiler/profile_gemm_b_scale_impl.hpp`  (+28/-15)
- `library/src/tensor_operation_instance/gpu/batched_gemm_b_scale/device_batched_gemm_b_scale_wmma_f16_i4_f16/device_batched_gemm_b_scale_wmma_f16_i4_f16_mk_nk_mn_mem_default_instance.cpp`  (+33/-0)
- `library/include/ck/library/tensor_operation_instance/gpu/batched_gemm_b_scale.hpp`  (+30/-0)
- `include/ck/library/utility/host_tensor_generator.hpp`  (+17/-1)
- `library/src/tensor_operation_instance/gpu/batched_gemm_b_scale/CMakeLists.txt`  (+4/-1)
- `test/batched_gemm_b_scale/CMakeLists.txt`  (+5/-0)
- `example/65_gemm_multiply_multiply/moe_gemm2_xdl_pk_i4.cpp`  (+2/-2)
- `profiler/src/CMakeLists.txt`  (+2/-2)

## Key added lines (kernel files)

**`example/65_gemm_multiply_multiply/moe_gemm2_xdl_pk_i4.cpp`**
```
b0_e_n_k.GenerateTensorValue(GeneratorTensor_3<B0DataType>{-1, 1});
b0_e_n_k.GenerateTensorValue(GeneratorTensor_3<B0DataType>{-1, 1});
```

**`include/ck/library/utility/host_tensor_generator.hpp`**
```
ck::pk_i4_t r = (((hi & 0xf) << 4) + (lo & 0xf));
template <>
struct GeneratorTensor_3<ck::pk_i4_t>
int min_value = 0;
```

**`include/ck/tensor_operation/gpu/device/impl/device_batched_gemm_wmma_cshuffle_v3_b_scale.hpp`**
```
namespace ck {
namespace tensor_operation {
namespace device {
template <typename GridwiseGemm,
```

**`include/ck/tensor_operation/gpu/grid/gridwise_gemm_wmma_cshuffle_v3_b_scale.hpp`**
```
using AsDataType_ = AsDataType;
using BsDataType_ = BsDataType;
```

**`library/include/ck/library/reference_tensor_operation/cpu/reference_batched_gemm.hpp`**
```
CElementwiseOperation c_element_op,
const int k_batch = 1)
c_element_op_{c_element_op},
k_batch_(k_batch)
```
