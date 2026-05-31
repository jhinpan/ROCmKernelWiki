# Diff summary

- **files changed:** 16
- **lines:** +794 / -28
- **kernel-ish files:** 14

## Files (by churn)

- `example/65_gemm_multiply_multiply/gemm_multiply_multiply_xdl_int8.cpp`  (+304/-0)
- `library/include/ck/library/tensor_operation_instance/gpu/gemm_multiply_multiply.hpp`  (+105/-0)
- `library/src/tensor_operation_instance/gpu/gemm_multiply_multiply/device_gemm_multiply_multiply_xdl_i8_i8_bf16/device_gemm_multiply_multiply_xdl_i8_i8_bf16_mk_nk_mn.hpp`  (+99/-0)
- `include/ck/host_utility/flush_cache.hpp`  (+38/-17)
- `library/src/tensor_operation_instance/gpu/gemm_multiply_multiply/device_gemm_multiply_multiply_xdl_i8_i8_bf16/device_gemm_multiply_multiply_xdl_i8_i8_bf16_mk_nk_mn_mem_v1_default_instance.cpp`  (+33/-0)
- `library/src/tensor_operation_instance/gpu/gemm_multiply_multiply/device_gemm_multiply_multiply_xdl_i8_i8_bf16/device_gemm_multiply_multiply_xdl_i8_i8_bf16_mk_nk_mn_mem_v1_kpadding_instance.cpp`  (+33/-0)
- `library/src/tensor_operation_instance/gpu/gemm_multiply_multiply/device_gemm_multiply_multiply_xdl_i8_i8_bf16/device_gemm_multiply_multiply_xdl_i8_i8_bf16_mk_nk_mn_mem_v2_default_instance.cpp`  (+33/-0)
- `library/src/tensor_operation_instance/gpu/gemm_multiply_multiply/device_gemm_multiply_multiply_xdl_i8_i8_bf16/device_gemm_multiply_multiply_xdl_i8_i8_bf16_mk_nk_mn_mem_v2_kpadding_instance.cpp`  (+33/-0)
- `library/src/tensor_operation_instance/gpu/gemm_multiply_multiply/device_gemm_multiply_multiply_xdl_i8_i8_bf16/device_gemm_multiply_multiply_xdl_i8_i8_bf16_mk_nk_mn_comp_default_instance.cpp`  (+32/-0)
- `library/src/tensor_operation_instance/gpu/gemm_multiply_multiply/device_gemm_multiply_multiply_xdl_i8_i8_bf16/device_gemm_multiply_multiply_xdl_i8_i8_bf16_mk_nk_mn_comp_kpadding_instance.cpp`  (+32/-0)
- `include/ck/tensor_operation/gpu/element/element_wise_operation.hpp`  (+20/-0)
- `include/ck/utility/amd_xdlops.hpp`  (+6/-6)
- `library/src/tensor_operation_instance/gpu/gemm_multiply_multiply/CMakeLists.txt`  (+10/-0)
- `profiler/include/profiler/profile_gemm_multiply_multiply_impl.hpp`  (+6/-4)
- `profiler/src/profile_gemm_multiply_multiply.cpp`  (+9/-1)

## Key added lines (kernel files)

**`example/65_gemm_multiply_multiply/gemm_multiply_multiply_xdl_int8.cpp`**
```
template <ck::index_t... Is>
using S = ck::Sequence<Is...>;
using I8  = int8_t;
using I32 = int;
```

**`include/ck/host_utility/flush_cache.hpp`**
```
hipEvent_t start, stop;
hip_check_error(hipEventCreate(&start));
hip_check_error(hipEventCreate(&stop));
hip_check_error(hipDeviceSynchronize());
```

**`include/ck/tensor_operation/gpu/element/element_wise_operation.hpp`**
```
template <>
__host__ __device__ constexpr void operator()<ck::half_t, int, ck::half_t, ck::half_t>(
ck::half_t& e, const int& c, const ck::half_t& d0, const ck::half_t& d1) const
const float x0_f =
```

**`include/ck/utility/amd_xdlops.hpp`**
```
__builtin_amdgcn_mfma_i32_16x16x32_i8(bit_cast<int64_t>(reg_a),
bit_cast<int64_t>(reg_b),
reg_c.template AsType<int32x4_t>()[Number<0>{}],
```

**`library/include/ck/library/tensor_operation_instance/gpu/gemm_multiply_multiply.hpp`**
```
void add_device_gemm_multiply_multiply_xdl_i8_i8_bf16_mk_nk_mn_comp_default_instances(
std::vector<std::unique_ptr<DeviceGemmMultipleDSplitK<Row,
Tuple<Row, Col>,
Tuple<F32, F32>,
```
