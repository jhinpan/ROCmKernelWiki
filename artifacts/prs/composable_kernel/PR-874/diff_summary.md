# Diff summary

- **files changed:** 17
- **lines:** +1749 / -28
- **kernel-ish files:** 14

## Files (by churn)

- `example/15_grouped_gemm/grouped_gemm_xdl_fixed_nk_fp8.cpp`  (+330/-0)
- `client_example/22_grouped_gemm/grouped_gemm_fixed_nk_fp16.cpp`  (+238/-0)
- `client_example/22_grouped_gemm/grouped_gemm_fixed_nk_fp8.cpp`  (+238/-0)
- `client_example/22_grouped_gemm/grouped_gemm_fixed_nk_i8.cpp`  (+238/-0)
- `library/include/ck/library/tensor_operation_instance/gpu/grouped_gemm_fixed_nk.hpp`  (+190/-0)
- `library/src/tensor_operation_instance/gpu/grouped_gemm_fixed_nk/device_grouped_gemm_xdl_fixed_nk_f16_f16_f16_mk_nk_mn_instance.cpp`  (+78/-0)
- `library/src/tensor_operation_instance/gpu/grouped_gemm_fixed_nk/device_grouped_gemm_xdl_fixed_nk_f16_f8_f16_mk_nk_mn_instance.cpp`  (+78/-0)
- `library/src/tensor_operation_instance/gpu/grouped_gemm_fixed_nk/device_grouped_gemm_xdl_fixed_nk_f16_i8_f16_mk_nk_mn_instance.cpp`  (+78/-0)
- `library/src/tensor_operation_instance/gpu/grouped_gemm_fixed_nk/device_grouped_gemm_xdl_fixed_nk_f16_f16_f16_mk_kn_mn_instance.cpp`  (+75/-0)
- `library/src/tensor_operation_instance/gpu/grouped_gemm_fixed_nk/device_grouped_gemm_xdl_fixed_nk_f16_f8_f16_mk_kn_mn_instance.cpp`  (+75/-0)
- `library/src/tensor_operation_instance/gpu/grouped_gemm_fixed_nk/device_grouped_gemm_xdl_fixed_nk_f16_i8_f16_mk_kn_mn_instance.cpp`  (+75/-0)
- `include/ck/tensor_operation/gpu/grid/gridwise_gemm_multiple_d_xdl_splitk_cshuffle.hpp`  (+18/-28)
- `include/ck/tensor_operation/gpu/element/unary_element_wise_operation.hpp`  (+12/-0)
- `library/src/tensor_operation_instance/gpu/grouped_gemm_fixed_nk/CMakeLists.txt`  (+10/-0)
- `client_example/22_grouped_gemm/CMakeLists.txt`  (+8/-0)

## Key added lines (kernel files)

**`client_example/22_grouped_gemm/grouped_gemm_fixed_nk_fp16.cpp`**
```
using F16 = ck::half_t;
using F32 = float;
using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;
```

**`client_example/22_grouped_gemm/grouped_gemm_fixed_nk_fp8.cpp`**
```
using F8  = ck::f8_t;
using F16 = ck::half_t;
using F32 = float;
using Row = ck::tensor_layout::gemm::RowMajor;
```

**`client_example/22_grouped_gemm/grouped_gemm_fixed_nk_i8.cpp`**
```
using I8  = int8_t;
using F16 = ck::half_t;
using F32 = float;
using Row = ck::tensor_layout::gemm::RowMajor;
```

**`example/15_grouped_gemm/grouped_gemm_xdl_fixed_nk_fp8.cpp`**
```
template <ck::index_t... Is>
using S = ck::Sequence<Is...>;
using F8  = ck::f8_t;
using F16 = ck::half_t;
```

**`include/ck/tensor_operation/gpu/device/impl/device_grouped_gemm_xdl_fixed_nk.hpp`**
```
typename ComputeType    = ADataType,
BDataType,
ComputeType,
```
