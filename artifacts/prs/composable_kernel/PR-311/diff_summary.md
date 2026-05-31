# Diff summary

- **files changed:** 113 (diff was byte-capped; summary is partial)
- **lines:** +1754 / -1080
- **kernel-ish files:** 108

## Files (by churn)

- `library/include/ck/library/tensor_operation_instance/gpu/gemm.hpp`  (+383/-0)
- `library/include/ck/library/tensor_operation_instance/gpu/device_gemm_instance.hpp`  (+0/-286)
- `library/include/ck/library/tensor_operation_instance/gpu/batched_gemm.hpp`  (+259/-0)
- `client_example/01_gemm/gemm.cpp`  (+218/-0)
- `library/include/ck/library/tensor_operation_instance/gpu/device_batched_gemm_instance.hpp`  (+0/-203)
- `library/include/ck/library/tensor_operation_instance/gpu/gemm_splitk.hpp`  (+147/-0)
- `library/include/ck/library/tensor_operation_instance/gpu/gemm_add_add_fastgelu.hpp`  (+141/-0)
- `library/include/ck/library/tensor_operation_instance/gpu/device_gemm_splitk_instance.hpp`  (+0/-124)
- `library/include/ck/library/tensor_operation_instance/gpu/device_gemm_add_add_fastgelu_instance.hpp`  (+0/-93)
- `client_example/02_gemm_add_add_fastgelu/gemm_add_add_fastgelu.cpp`  (+33/-29)
- `include/ck/tensor_operation/gpu/device/device_gemm.hpp`  (+36/-17)
- `library/include/ck/library/utility/conv_util.hpp`  (+19/-19)
- `library/src/tensor_operation_instance/gpu/batched_gemm/device_batched_gemm_xdl_bf16_bf16_bf16_gkm_gkn_gmn_instance.cpp`  (+19/-16)
- `library/include/ck/library/tensor_operation_instance/device_operation_instance_factory.hpp`  (+33/-0)
- `include/ck/tensor_operation/gpu/device/device_gemm_multiple_d.hpp`  (+25/-5)

## Key added lines (kernel files)

**`client_example/01_gemm/gemm.cpp`**
```
using F16 = ck::half_t;
using F32 = float;
using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;
```

**`client_example/02_gemm_add_add_fastgelu/gemm_add_add_fastgelu.cpp`**
```
using ADataType  = F16;
using BDataType  = F16;
using D0DataType = F16;
using D1DataType = F16;
```

**`client_example/03_gemm_layernorm/gemm_add_add_layernorm.cpp`**
```
const auto gemm_reduce_ptrs =
ck::tensor_operation::device::instance::get_device_gemm_add_add_mean_squaremean_instances<
ADataType,
BDataType,
```

**`include/ck/tensor_operation/gpu/device/device_batched_gemm.hpp`**
```
template <typename ALayout,
typename BLayout,
typename CLayout,
typename ADataType,
```

**`include/ck/tensor_operation/gpu/device/device_batched_gemm_xdl.hpp`**
```
struct DeviceBatchedGemmXdl : public DeviceBatchedGemm<ALayout,
ADataType,
BDataType,
CDataType,
```
