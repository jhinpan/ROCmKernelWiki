# Diff summary

- **files changed:** 46 (diff was byte-capped; summary is partial)
- **lines:** +3698 / -331
- **kernel-ish files:** 45

## Files (by churn)

- `include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_b_preshuffle_v3.hpp`  (+860/-0)
- `include/ck/tensor_operation/gpu/device/impl/device_gemm_multiple_d_xdl_cshuffle_v3_b_preshuffle.hpp`  (+581/-0)
- `include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_b_preshuffle_v2.hpp`  (+558/-0)
- `include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_b_preshuffle_v1.hpp`  (+506/-0)
- `example/65_gemm_multiply_multiply/gemm_multiply_multiply_xdl_fp8_bpreshuffle.cpp`  (+396/-0)
- `example/ck_tile/11_add_rmsnorm2d_rdquant/instances/add_rmsnorm2d_rdquant_fwd_api.cpp`  (+126/-49)
- `example/ck_tile/11_add_rmsnorm2d_rdquant/add_rmsnorm2d_rdquant_fwd.cpp`  (+70/-43)
- `include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_v4.hpp`  (+47/-66)
- `include/ck/tensor_operation/gpu/block/blockwise_gemm_pipeline_xdlops_b_preshuffle_selector.hpp`  (+110/-0)
- `example/65_gemm_multiply_multiply/gemm_multiply_multiply_xdl_int8.cpp`  (+43/-6)
- `include/ck/tensor_operation/gpu/device/device_gemm_multiple_d.hpp`  (+45/-0)
- `example/ck_tile/11_add_rmsnorm2d_rdquant/instances/add_rmsnorm2d_rdquant_fwd_bf16_n8192_instance.cpp`  (+42/-0)
- `example/ck_tile/11_add_rmsnorm2d_rdquant/instances/add_rmsnorm2d_rdquant_fwd_fp16_n8192_instance.cpp`  (+41/-0)
- `example/ck_tile/11_add_rmsnorm2d_rdquant/add_rmsnorm2d_rdquant_fwd.hpp`  (+33/-6)
- `example/65_gemm_multiply_multiply/gemm_multiply_multiply_xdl_fp8.cpp`  (+13/-10)

## Key added lines (kernel files)

**`example/65_gemm_multiply_multiply/gemm_multiply_multiply_xdl_fp8.cpp`**
```
static constexpr auto GemmSpec = ck::tensor_operation::device::GemmSpecialization::Default;
<Row, Col, DsLayout, ELayout,
A0DataType, B0DataType, DsDataType, EDataType, AccDataType, CShuffleDataType,
AElementOp,  BElementOp, CDEElementOp,       GemmSpec,   256,
```

**`example/65_gemm_multiply_multiply/gemm_multiply_multiply_xdl_fp8_bpreshuffle.cpp`**
```
template <ck::index_t... Is>
using S = ck::Sequence<Is...>;
using F16  = ck::half_t;
using BF16 = ck::bhalf_t;
```

**`example/65_gemm_multiply_multiply/gemm_multiply_multiply_xdl_int8.cpp`**
```
using CShuffleDataType = F16;
using D0DataType       = F16;
using D1DataType       = F16;
template <>
```

**`example/ck_tile/11_add_rmsnorm2d_rdquant/add_rmsnorm2d_rdquant_fwd.cpp`**
```
template <typename InputDataType>
.insert("quant", "int8", "precision")
template <typename InputDataType, typename QuantizedDataType, bool SaveX>
float epsilon                   = arg_parser.get_float("e");
```

**`example/ck_tile/11_add_rmsnorm2d_rdquant/add_rmsnorm2d_rdquant_fwd.hpp`**
```
template <typename InputDataType, typename QuantizedDataType>
struct AddRmsnormRdquantTypeConfig<ck_tile::half_t, ck_tile::int8_t>
struct AddRmsnormRdquantTypeConfig<ck_tile::bf16_t, ck_tile::int8_t>
template <>
```
