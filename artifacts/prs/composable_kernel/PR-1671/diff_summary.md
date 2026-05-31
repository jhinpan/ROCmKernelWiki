# Diff summary

- **files changed:** 16
- **lines:** +896 / -26
- **kernel-ish files:** 14

## Files (by churn)

- `example/01_gemm/gemm_wmma_bf16.cpp`  (+84/-0)
- `example/01_gemm/gemm_wmma_int8.cpp`  (+84/-0)
- `library/src/tensor_operation_instance/gpu/gemm/device_gemm_wmma_bf16_bf16_bf16_km_kn_mn_instance.cpp`  (+77/-0)
- `library/src/tensor_operation_instance/gpu/gemm/device_gemm_wmma_bf16_bf16_bf16_km_nk_mn_instance.cpp`  (+77/-0)
- `library/src/tensor_operation_instance/gpu/gemm/device_gemm_wmma_bf16_bf16_bf16_mk_kn_mn_instance.cpp`  (+77/-0)
- `library/src/tensor_operation_instance/gpu/gemm/device_gemm_wmma_bf16_bf16_bf16_mk_nk_mn_instance.cpp`  (+77/-0)
- `library/src/tensor_operation_instance/gpu/gemm/device_gemm_wmma_int8_int8_int8_km_kn_mn_instance.cpp`  (+76/-0)
- `library/src/tensor_operation_instance/gpu/gemm/device_gemm_wmma_int8_int8_int8_km_nk_mn_instance.cpp`  (+76/-0)
- `library/src/tensor_operation_instance/gpu/gemm/device_gemm_wmma_int8_int8_int8_mk_kn_mn_instance.cpp`  (+76/-0)
- `library/src/tensor_operation_instance/gpu/gemm/device_gemm_wmma_int8_int8_int8_mk_nk_mn_instance.cpp`  (+76/-0)
- `library/include/ck/library/tensor_operation_instance/gpu/gemm.hpp`  (+52/-0)
- `library/include/ck/library/tensor_operation_instance/gpu/gemm_wmma.inc`  (+40/-0)
- `library/src/tensor_operation_instance/gpu/gemm/CMakeLists.txt`  (+13/-20)
- `include/ck/utility/amd_wmma.hpp`  (+6/-5)
- `example/01_gemm/CMakeLists.txt`  (+4/-0)

## Key added lines (kernel files)

**`example/01_gemm/gemm_wmma_bf16.cpp`**
```
using ADataType        = ck::bhalf_t;
using BDataType        = ck::bhalf_t;
using AccDataType      = float;
using CShuffleDataType = float;
```

**`example/01_gemm/gemm_wmma_int8.cpp`**
```
using ADataType        = int8_t;
using BDataType        = int8_t;
using AccDataType      = int32_t;
using CShuffleDataType = int32_t;
```

**`library/include/ck/library/tensor_operation_instance/gpu/gemm.hpp`**
```
if constexpr(is_same_v<ADataType, ck::bhalf_t> && is_same_v<BDataType, ck::bhalf_t> &&
is_same_v<CDataType, ck::bhalf_t>)
if constexpr(is_same_v<ALayout, Row> && is_same_v<BLayout, Row> &&
is_same_v<CLayout, Row>)
```

**`library/include/ck/library/tensor_operation_instance/gpu/gemm_wmma.inc`**
```
void add_device_gemm_wmma_bf16_bf16_bf16_km_kn_mn_instances(
std::vector<std::unique_ptr<
DeviceGemm<Col, Row, Row, BF16, BF16, BF16, PassThrough, PassThrough, PassThrough>>>&
instances);
```

**`library/include/ck/library/utility/check_err.hpp`**
```
double rtol            = 1e-1,
```
