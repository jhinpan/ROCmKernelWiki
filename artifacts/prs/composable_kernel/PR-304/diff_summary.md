# Diff summary

- **files changed:** 68 (diff was byte-capped; summary is partial)
- **lines:** +2101 / -1878
- **kernel-ish files:** 59

## Files (by churn)

- `profiler/include/profile_gemm_impl.hpp`  (+70/-460)
- `profiler/src/profile_gemm.cpp`  (+87/-317)
- `profiler/src/profile_batched_gemm.cpp`  (+75/-292)
- `profiler/include/profile_batched_gemm_impl.hpp`  (+58/-271)
- `library/include/ck/library/tensor_operation_instance/gpu/device_gemm_instance.hpp`  (+286/-0)
- `profiler/include/profile_gemm_splitk_impl.hpp`  (+256/-0)
- `client_example/02_gemm_add_add_fastgelu/gemm_add_add_fastgelu.cpp`  (+237/-0)
- `library/include/ck/library/tensor_operation_instance/gpu/device_batched_gemm_instance.hpp`  (+203/-0)
- `profiler/src/profile_gemm_splitk.cpp`  (+148/-0)
- `profiler/include/profile_gemm_add_add_fastgelu_impl.hpp`  (+45/-94)
- `library/include/ck/library/tensor_operation_instance/gpu/device_gemm_splitk_instance.hpp`  (+124/-0)
- `test/batched_gemm/batched_gemm_util.hpp`  (+0/-109)
- `library/include/ck/library/tensor_operation_instance/gpu/device_gemm_add_add_fastgelu_instance.hpp`  (+93/-0)
- `test/gemm/gemm_util.hpp`  (+5/-87)
- `include/ck/tensor_operation/gpu/device/device_batched_gemm_reduce.hpp`  (+54/-0)

## Key added lines (kernel files)

**`client_example/02_gemm_add_add_fastgelu/gemm_add_add_fastgelu.cpp`**
```
using F16 = ck::half_t;
using F32 = float;
using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;
```

**`example/01_gemm/gemm_xdl_bf16.cpp`**
```
using ReferenceGemmInstance = ck::tensor_operation::host::ReferenceGemm<ADataType,
BDataType,
CDataType,
AccDataType,
```

**`include/ck/tensor_operation/gpu/device/device_batched_gemm.hpp`**
```
namespace ck {
namespace tensor_operation {
namespace device {
template <typename AElementwiseOperation,
```

**`include/ck/tensor_operation/gpu/device/device_batched_gemm_reduce.hpp`**
```
namespace ck {
namespace tensor_operation {
namespace device {
template <typename AElementwiseOperation,
```

**`include/ck/tensor_operation/gpu/device/device_batched_gemm_reduce_xdl_cshuffle.hpp`**
```
: public DeviceBatchedGemmReduce<AElementwiseOperation,
BElementwiseOperation,
CElementwiseOperation,
DxsInElementwiseOperation,
```
