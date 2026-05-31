# Diff summary

- **files changed:** 15
- **lines:** +758 / -13
- **kernel-ish files:** 14

## Files (by churn)

- `library/include/ck/library/tensor_operation_instance/gpu/gemm_ab_scale.hpp`  (+173/-0)
- `library/src/tensor_operation_instance/gpu/gemm_ab_scale/device_gemm_ab_scale_xdl_f8_f8_bf16/device_gemm_ab_scale_xdl_f8_f8_bf16_km_kn_mn_128_128_128.hpp`  (+96/-0)
- `library/src/tensor_operation_instance/gpu/gemm_ab_scale/device_gemm_ab_scale_xdl_f8_f8_bf16/device_gemm_ab_scale_xdl_f8_f8_bf16_mk_kn_mn_128_128_128.hpp`  (+87/-0)
- `include/ck/tensor_operation/gpu/grid/gridwise_gemm_xdl_cshuffle_v3_multi_d_ab_scale.hpp`  (+30/-8)
- `library/src/tensor_operation_instance/gpu/gemm_ab_scale/device_gemm_ab_scale_xdl_f8_f8_bf16/device_gemm_ab_scale_xdl_f8_f8_bf16_km_kn_mn_128_128_128_mem_v1_default_instance.cpp`  (+38/-0)
- `library/src/tensor_operation_instance/gpu/gemm_ab_scale/device_gemm_ab_scale_xdl_f8_f8_bf16/device_gemm_ab_scale_xdl_f8_f8_bf16_km_kn_mn_128_128_128_mem_v1_kpadding_instance.cpp`  (+38/-0)
- `library/src/tensor_operation_instance/gpu/gemm_ab_scale/device_gemm_ab_scale_xdl_f8_f8_bf16/device_gemm_ab_scale_xdl_f8_f8_bf16_mk_kn_mn_128_128_128_mem_v1_default_instance.cpp`  (+38/-0)
- `library/src/tensor_operation_instance/gpu/gemm_ab_scale/device_gemm_ab_scale_xdl_f8_f8_bf16/device_gemm_ab_scale_xdl_f8_f8_bf16_mk_kn_mn_128_128_128_mem_v1_kpadding_instance.cpp`  (+38/-0)
- `library/src/tensor_operation_instance/gpu/gemm_ab_scale/device_gemm_ab_scale_xdl_f8_f8_bf16/device_gemm_ab_scale_xdl_f8_f8_bf16_km_kn_mn_128_128_128_comp_default_instance.cpp`  (+37/-0)
- `library/src/tensor_operation_instance/gpu/gemm_ab_scale/device_gemm_ab_scale_xdl_f8_f8_bf16/device_gemm_ab_scale_xdl_f8_f8_bf16_km_kn_mn_128_128_128_comp_kpadding_instance.cpp`  (+37/-0)
- `library/src/tensor_operation_instance/gpu/gemm_ab_scale/device_gemm_ab_scale_xdl_f8_f8_bf16/device_gemm_ab_scale_xdl_f8_f8_bf16_mk_kn_mn_128_128_128_comp_default_instance.cpp`  (+37/-0)
- `library/src/tensor_operation_instance/gpu/gemm_ab_scale/device_gemm_ab_scale_xdl_f8_f8_bf16/device_gemm_ab_scale_xdl_f8_f8_bf16_mk_kn_mn_128_128_128_comp_kpadding_instance.cpp`  (+37/-0)
- `profiler/src/profile_gemm_ab_scale.cpp`  (+34/-0)
- `library/src/tensor_operation_instance/gpu/gemm_ab_scale/CMakeLists.txt`  (+22/-0)
- `include/ck/tensor_operation/gpu/device/impl/device_gemm_multiple_d_xdl_cshuffle_v3_ab_scale.hpp`  (+16/-5)

## Key added lines (kernel files)

**`include/ck/tensor_operation/gpu/device/impl/device_gemm_multiple_d_xdl_cshuffle_v3_ab_scale.hpp`**
```
constexpr index_t minimum_occupancy = [&]() {
if constexpr(is_same_v<tensor_layout::gemm::ColumnMajor, ALayout> &&
is_same_v<tensor_layout::gemm::RowMajor, BLayout>)
return 1;
```

**`include/ck/tensor_operation/gpu/grid/gridwise_gemm_xdl_cshuffle_v3_multi_d_ab_scale.hpp`**
```
__host__ __device__ static constexpr auto MakeAScaleGridDesciptor_M_K(index_t M, index_t K)
const auto BM = math::integer_divide_ceil(M, ScaleBlockM);
const auto BK = math::integer_divide_ceil(K, ScaleBlockK);
if constexpr(is_same<tensor_layout::gemm::RowMajor, ALayout>::value)
```

**`library/include/ck/library/tensor_operation_instance/gpu/gemm_ab_scale.hpp`**
```
void add_device_gemm_ab_scale_xdl_f8_f8_bf16_mk_kn_mn_1_128_128_comp_default_instances(
std::vector<std::unique_ptr<DeviceGemmMultipleD_ABScale<Row,
PassThrough,
PassThrough,
```

**`library/src/tensor_operation_instance/gpu/gemm_ab_scale/device_gemm_ab_scale_xdl_f8_f8_bf16/device_gemm_ab_scale_xdl_f8_f8_bf16_km_kn_mn_128_128_128.hpp`**
```
namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {
```

**`library/src/tensor_operation_instance/gpu/gemm_ab_scale/device_gemm_ab_scale_xdl_f8_f8_bf16/device_gemm_ab_scale_xdl_f8_f8_bf16_km_kn_mn_128_128_128_comp_default_instance.cpp`**
```
namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {
```
