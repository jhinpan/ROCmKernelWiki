# Diff summary

- **files changed:** 10
- **lines:** +398 / -31
- **kernel-ish files:** 9

## Files (by churn)

- `library/src/tensor_operation_instance/gpu/gemm_splitk/device_gemm_xdl_splitk_f16_f16_f16_comp_fp8_mk_kn_mn_instance.cpp`  (+83/-0)
- `library/src/tensor_operation_instance/gpu/gemm_splitk/device_gemm_xdl_splitk_f16_f16_f16_comp_fp8_mk_nk_mn_instance.cpp`  (+79/-0)
- `library/src/tensor_operation_instance/gpu/gemm_splitk/device_gemm_xdl_splitk_f16_f16_f16_comp_fp8_km_kn_mn_instance.cpp`  (+62/-0)
- `library/src/tensor_operation_instance/gpu/gemm_splitk/device_gemm_xdl_splitk_f16_f16_f16_comp_fp8_km_nk_mn_instance.cpp`  (+62/-0)
- `profiler/src/profile_gemm_splitk.cpp`  (+41/-19)
- `library/include/ck/library/tensor_operation_instance/gpu/gemm_splitk.hpp`  (+52/-4)
- `include/ck/tensor_operation/gpu/device/device_gemm_splitk.hpp`  (+6/-3)
- `profiler/include/profiler/profile_gemm_splitk_impl.hpp`  (+6/-3)
- `library/src/tensor_operation_instance/gpu/gemm_splitk/CMakeLists.txt`  (+5/-1)
- `include/ck/tensor_operation/gpu/device/impl/device_gemm_xdl_splitk_c_shuffle.hpp`  (+2/-1)

## Key added lines (kernel files)

**`include/ck/tensor_operation/gpu/device/device_gemm_splitk.hpp`**
```
typename CElementwiseOperation,
typename ComputeType = CDataType>
typename CElementwiseOperation,
typename ComputeType = CDataType>
```

**`include/ck/tensor_operation/gpu/device/impl/device_gemm_xdl_splitk_c_shuffle.hpp`**
```
CElementwiseOperation,
ComputeType>
```

**`library/include/ck/library/tensor_operation_instance/gpu/gemm_splitk.hpp`**
```
void add_device_gemm_xdl_splitk_f16_f16_f16_comp_f8_km_kn_mn_instances(
std::vector<std::unique_ptr<
DeviceGemmSplitK<Col, Row, Row, F16, F16, F16, PassThrough, PassThrough, PassThrough, F8>>>&
instances);
```

**`library/src/tensor_operation_instance/gpu/gemm_splitk/device_gemm_xdl_splitk_f16_f16_f16_comp_fp8_km_kn_mn_instance.cpp`**
```
namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {
```

**`library/src/tensor_operation_instance/gpu/gemm_splitk/device_gemm_xdl_splitk_f16_f16_f16_comp_fp8_km_nk_mn_instance.cpp`**
```
namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {
```
