# Diff summary

- **files changed:** 8
- **lines:** +203 / -68
- **kernel-ish files:** 8

## Files (by churn)

- `include/ck/tensor_operation/gpu/grid/gridwise_moe_gemm.hpp`  (+54/-39)
- `example/65_gemm_multiply_multiply/moe_gemm1_xdl_fp8.cpp`  (+54/-10)
- `example/65_gemm_multiply_multiply/moe_gemm1_xdl_pk_i4.cpp`  (+54/-9)
- `library/include/ck/library/reference_tensor_operation/cpu/reference_moe_gemm.hpp`  (+14/-1)
- `library/include/ck/library/reference_tensor_operation/cpu/reference_moe_gemm2.hpp`  (+10/-2)
- `example/65_gemm_multiply_multiply/moe_gemm2_xdl_fp8.cpp`  (+5/-3)
- `example/65_gemm_multiply_multiply/moe_gemm2_xdl_pk_i4.cpp`  (+5/-3)
- `include/ck/tensor_operation/gpu/device/impl/device_moe_gemm.hpp`  (+7/-1)

## Key added lines (kernel files)

**`example/65_gemm_multiply_multiply/moe_gemm1_xdl_fp8.cpp`**
```
using D2DataType       = F32;
using DsDataType       = ck::Tuple<D0DataType, D1DataType, D2DataType>;
using D2Layout = ELayout;
using DsLayout = ck::Tuple<D0Layout, D1Layout, D2Layout>;
```

**`example/65_gemm_multiply_multiply/moe_gemm1_xdl_pk_i4.cpp`**
```
using D2DataType       = F32;
using DsDataType       = ck::Tuple<D0DataType, D1DataType, D2DataType>;
using DsLayout = ck::Tuple<D0Layout, D1Layout, ELayout>;
struct MulABScaleExpertWeight
```

**`example/65_gemm_multiply_multiply/moe_gemm2_xdl_fp8.cpp`**
```
static constexpr bool MulRoutedWeight      = false;
ck::BlockGemmPipelineScheduler::Intrawave, ck::BlockGemmPipelineVersion::v1, false, false, MulRoutedWeight, A0DataType>;
CDEElementOp,
MulRoutedWeight>;
```

**`example/65_gemm_multiply_multiply/moe_gemm2_xdl_pk_i4.cpp`**
```
static constexpr bool MulRoutedWeight      = true;
ck::BlockGemmPipelineScheduler::Intrawave, ck::BlockGemmPipelineVersion::v1, false, false, MulRoutedWeight, A0DataType>;
CDEElementOp,
MulRoutedWeight>;
```

**`include/ck/tensor_operation/gpu/device/impl/device_moe_gemm.hpp`**
```
bool MulRoutedWeight                        = true,
MulRoutedWeight,
MulRoutedWeight,
MulRoutedWeight,
```
