# Diff summary

- **files changed:** 38
- **lines:** +1739 / -39
- **kernel-ish files:** 33

## Files (by churn)

- `test/gemm_universal_streamk/test_gemm_universal_streamk_ut_cases_bf16.inc`  (+177/-0)
- `library/include/ck/library/tensor_operation_instance/gpu/gemm_universal_streamk.hpp`  (+127/-2)
- `test/gemm_universal_streamk/test_gemm_universal_streamk_ut_cases_fp16.inc`  (+113/-0)
- `test/gemm_universal_streamk/test_gemm_universal_streamk_ut_cases_fp8.inc`  (+113/-0)
- `library/src/tensor_operation_instance/gpu/gemm_universal_streamk/device_gemm_xdl_universal_streamk_f8_f8_bf16/device_gemm_xdl_universal_streamk_f8_f8_bf16_mk_nk_mn.hpp`  (+107/-0)
- `test/gemm_universal_streamk/test_gemm_universal_streamk_util.hpp`  (+104/-0)
- `library/src/tensor_operation_instance/gpu/gemm_universal_streamk/device_gemm_xdl_universal_streamk_f8_f8_bf16/device_gemm_xdl_universal_streamk_f8_f8_bf16_mk_kn_mn.hpp`  (+99/-0)
- `test/gemm_universal_streamk/test_gemm_universal_streamk_xdl_bf16.cpp`  (+85/-0)
- `test/gemm_universal_streamk/test_gemm_universal_streamk_xdl_fp16.cpp`  (+84/-0)
- `test/gemm_universal_streamk/test_gemm_universal_streamk_xdl_fp8.cpp`  (+74/-0)
- `example/01_gemm/gemm_xdl_fp16_fp8_streamk_v3.cpp`  (+64/-0)
- `profiler/src/profile_gemm_universal_streamk.cpp`  (+29/-15)
- `profiler/include/profiler/profile_gemm_universal_streamk_impl.hpp`  (+25/-15)
- `library/src/tensor_operation_instance/gpu/gemm_universal_streamk/device_gemm_xdl_universal_streamk_bf16_bf16_bf16/device_gemm_xdl_universal_streamk_bf16_bf16_bf16_mk_nk_mn_comp_mnkpadding_instance.cpp`  (+31/-0)
- `library/src/tensor_operation_instance/gpu/gemm_universal_streamk/device_gemm_xdl_universal_streamk_bf16_bf16_bf16/device_gemm_xdl_universal_streamk_bf16_bf16_bf16_mk_nk_mn_mem_v1_mnkpadding_instance.cpp`  (+31/-0)

## Key added lines (kernel files)

**`example/01_gemm/gemm_xdl_fp16_fp8_streamk_v3.cpp`**
```
using ADataType        = ck::half_t;
using BDataType        = ck::f8_t;
using AccDataType      = float;
using CShuffleDataType = ck::half_t;
```

**`library/include/ck/library/tensor_operation_instance/gpu/gemm_universal_streamk.hpp`**
```
void add_device_gemm_xdl_universal_streamk_f8_f8_bf16_mk_kn_mn_comp_default_instances(
std::vector<std::unique_ptr<
DeviceGemm_Streamk_V2<Row, Row, Row, F8, F8, BF16, PassThrough, PassThrough, PassThrough>>>&
instances);
```

**`library/src/tensor_operation_instance/gpu/gemm_universal_streamk/device_gemm_xdl_universal_streamk_bf16_bf16_bf16/device_gemm_xdl_universal_streamk_bf16_bf16_bf16_mk_nk_mn_comp_mnkpadding_instance.cpp`**
```
namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {
```

**`library/src/tensor_operation_instance/gpu/gemm_universal_streamk/device_gemm_xdl_universal_streamk_bf16_bf16_bf16/device_gemm_xdl_universal_streamk_bf16_bf16_bf16_mk_nk_mn_comp_mnpadding_instance.cpp`**
```
namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {
```

**`library/src/tensor_operation_instance/gpu/gemm_universal_streamk/device_gemm_xdl_universal_streamk_bf16_bf16_bf16/device_gemm_xdl_universal_streamk_bf16_bf16_bf16_mk_nk_mn_mem_v1_mnkpadding_instance.cpp`**
```
namespace ck {
namespace tensor_operation {
namespace device {
namespace instance {
```
