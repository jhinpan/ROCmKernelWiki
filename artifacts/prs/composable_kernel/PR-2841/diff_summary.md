# Diff summary

- **files changed:** 21
- **lines:** +1166 / -7
- **kernel-ish files:** 15

## Files (by churn)

- `profiler/include/profiler/profile_gemm_quantization_impl.hpp`  (+231/-0)
- `example/14_gemm_quantization/gemm_wmma_quantization_int8.cpp`  (+211/-0)
- `library/include/ck/library/tensor_operation_instance/gpu/quantization/gemm_quantization.hpp`  (+175/-3)
- `profiler/src/profile_gemm_quantization.cpp`  (+115/-0)
- `library/src/tensor_operation_instance/gpu/quantization/gemm/device_gemm_quantization_wmma_c_shuffle_i8_i8_i8_instance.hpp`  (+79/-0)
- `test/quantization/gemm/test_gemm_quantization_util.hpp`  (+62/-0)
- `library/src/tensor_operation_instance/gpu/quantization/gemm/device_gemm_quantization_wmma_c_shuffle_i8_i8_i8_km_kn_mn_instance.cpp`  (+41/-0)
- `library/src/tensor_operation_instance/gpu/quantization/gemm/device_gemm_quantization_wmma_c_shuffle_i8_i8_i8_km_nk_mn_instance.cpp`  (+41/-0)
- `library/src/tensor_operation_instance/gpu/quantization/gemm/device_gemm_quantization_wmma_c_shuffle_i8_i8_i8_mk_kn_mn_instance.cpp`  (+41/-0)
- `library/src/tensor_operation_instance/gpu/quantization/gemm/device_gemm_quantization_wmma_c_shuffle_i8_i8_i8_mk_nk_mn_instance.cpp`  (+41/-0)
- `test/quantization/gemm/test_gemm_quantization_ut_cases.inc`  (+41/-0)
- `test/quantization/gemm/test_gemm_quantization.cpp`  (+40/-0)
- `include/ck/tensor_operation/gpu/grid/gridwise_gemm_wmma_cshuffle_v3_common.hpp`  (+14/-0)
- `profiler/src/CMakeLists.txt`  (+9/-0)
- `test/quantization/gemm/CMakeLists.txt`  (+9/-0)

## Key added lines (kernel files)

**`example/14_gemm_quantization/gemm_wmma_quantization_int8.cpp`**
```
template <ck::index_t... Is>
using S = ck::Sequence<Is...>;
using I8  = int8_t;
using I32 = int32_t;
```

**`include/ck/tensor_operation/gpu/device/impl/device_gemm_wmma_cshuffle_v3_common.hpp`**
```
!(std::is_same_v<EDataType, ck::half_t> || std::is_same_v<EDataType, ck::bhalf_t> ||
std::is_same_v<EDataType, int8_t>) ||
```

**`include/ck/tensor_operation/gpu/grid/gridwise_gemm_wmma_cshuffle_v3_common.hpp`**
```
if constexpr(is_same<remove_cvref_t<EDataType>, int8_t>::value)
if(karg.KBatch > 1)
if(ck::EnvIsEnabled(CK_ENV(CK_LOGGING)))
std::cout << "int8_t does not support KBatch > 1. KBatch: " << karg.KBatch
```

**`library/include/ck/library/tensor_operation_instance/gpu/quantization/gemm_quantization.hpp`**
```
void add_device_gemm_quantization_wmma_c_shuffle_i8_i8_i8_km_kn_mn_instances(
std::vector<std::unique_ptr<DeviceGemmMultipleDSplitK<Col,
Empty_Tuple,
Empty_Tuple,
```

**`library/src/tensor_operation_instance/gpu/quantization/gemm/device_gemm_quantization_wmma_c_shuffle_i8_i8_i8_instance.hpp`**
```
namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {
```
