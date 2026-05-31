# Diff summary

- **files changed:** 33
- **lines:** +2424 / -6
- **kernel-ish files:** 26

## Files (by churn)

- `profiler/include/profiler/profile_gemm_add_impl.hpp`  (+232/-0)
- `profiler/include/profiler/profile_gemm_add_relu_impl.hpp`  (+232/-0)
- `profiler/include/profiler/profile_gemm_add_silu_impl.hpp`  (+232/-0)
- `profiler/src/profile_gemm_add.cpp`  (+139/-0)
- `profiler/src/profile_gemm_add_relu.cpp`  (+139/-0)
- `profiler/src/profile_gemm_add_silu.cpp`  (+139/-0)
- `library/include/ck/library/tensor_operation_instance/gpu/gemm_add_relu.hpp`  (+116/-0)
- `library/include/ck/library/tensor_operation_instance/gpu/gemm_add_silu.hpp`  (+116/-0)
- `library/include/ck/library/tensor_operation_instance/gpu/gemm_add.hpp`  (+114/-0)
- `include/ck/tensor_operation/gpu/element/binary_element_wise_operation.hpp`  (+82/-1)
- `library/src/tensor_operation_instance/gpu/gemm_add_fastgelu/device_gemm_add_fastgelu_xdl_c_shuffle_bf16_i8_bf16_bf16_mk_kn_mn_mn_instance.cpp`  (+73/-0)
- `library/src/tensor_operation_instance/gpu/gemm_add_fastgelu/device_gemm_add_fastgelu_xdl_c_shuffle_f16_i8_f16_f16_mk_kn_mn_mn_instance.cpp`  (+72/-0)
- `test/gemm_add/test_gemm_add.hpp`  (+72/-0)
- `library/src/tensor_operation_instance/gpu/gemm_add_relu/device_gemm_add_relu_xdl_c_shuffle_bf16_i8_bf16_bf16_mk_kn_mn_mn_instance.cpp`  (+71/-0)
- `library/src/tensor_operation_instance/gpu/gemm_add_silu/device_gemm_add_silu_xdl_c_shuffle_bf16_i8_bf16_bf16_mk_kn_mn_mn_instance.cpp`  (+71/-0)

## Key added lines (kernel files)

**`include/ck/tensor_operation/gpu/element/binary_element_wise_operation.hpp`**
```
template <>
__host__ __device__ constexpr void
operator()<bhalf_t>(bhalf_t& y, const float& x0, const bhalf_t& x1) const
const float x2_tmp = ck::type_convert<float>(x1);
```

**`include/ck/tensor_operation/gpu/element/unary_element_wise_operation.hpp`**
```
template <>
__host__ __device__ void operator()<bhalf_t, int8_t>(bhalf_t& y, const int8_t& x) const
y = type_convert<bhalf_t>(x);
struct Silu
```

**`library/include/ck/library/tensor_operation_instance/device_operation_instance_factory.hpp`**
```
using AddRelu        = ck::tensor_operation::element_wise::AddRelu;
using AddSilu        = ck::tensor_operation::element_wise::AddSilu;
```

**`library/include/ck/library/tensor_operation_instance/gpu/gemm_add.hpp`**
```
namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {
```

**`library/include/ck/library/tensor_operation_instance/gpu/gemm_add_fastgelu.hpp`**
```
void add_device_gemm_add_fastgelu_xdl_c_shuffle_f16_i8_f16_f16_mk_kn_mn_mn_instances(
std::vector<std::unique_ptr<DeviceGemmMultipleD<Row,
Row_Tuple,
F16_Tuple,
```
