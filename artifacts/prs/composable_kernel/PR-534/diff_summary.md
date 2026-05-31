# Diff summary

- **files changed:** 20
- **lines:** +2338 / -0
- **kernel-ish files:** 15

## Files (by churn)

- `include/ck/tensor_operation/gpu/device/impl/device_gemm_multiple_d_dl.hpp`  (+669/-0)
- `profiler/include/profiler/profile_gemm_add_multiply_impl.hpp`  (+242/-0)
- `client_example/15_gemm_add_multiply/gemm_add_multiply.cpp`  (+241/-0)
- `profiler/src/profile_gemm_add_multiply.cpp`  (+158/-0)
- `library/include/ck/library/tensor_operation_instance/gpu/gemm_add_multiply.hpp`  (+155/-0)
- `library/src/tensor_operation_instance/gpu/gemm_add_multiply/device_gemm_add_multiply_xdl_c_shuffle_f16_f16_f16_f16_f16_mk_nk_mn_mn_mn_instance.cpp`  (+143/-0)
- `example/46_gemm_add_multiply/run_gemm_add_multiply_example.inc`  (+140/-0)
- `library/src/tensor_operation_instance/gpu/gemm_add_multiply/device_gemm_add_multiply_xdl_c_shuffle_f16_f16_f16_f16_f16_km_kn_mn_mn_mn_instance.cpp`  (+106/-0)
- `library/src/tensor_operation_instance/gpu/gemm_add_multiply/device_gemm_add_multiply_xdl_c_shuffle_f16_f16_f16_f16_f16_km_nk_mn_mn_mn_instance.cpp`  (+106/-0)
- `library/src/tensor_operation_instance/gpu/gemm_add_multiply/device_gemm_add_multiply_xdl_c_shuffle_f16_f16_f16_f16_f16_mk_kn_mn_mn_mn_instance.cpp`  (+106/-0)
- `example/46_gemm_add_multiply/common.hpp`  (+102/-0)
- `example/46_gemm_add_multiply/gemm_add_multiply_dl_fp16.cpp`  (+47/-0)
- `example/46_gemm_add_multiply/gemm_add_multiply_xdl_fp16.cpp`  (+47/-0)
- `include/ck/tensor_operation/gpu/element/element_wise_operation.hpp`  (+36/-0)
- `example/46_gemm_add_multiply/README.md`  (+26/-0)

## Key added lines (kernel files)

**`client_example/15_gemm_add_multiply/gemm_add_multiply.cpp`**
```
using F16 = ck::half_t;
using F32 = float;
using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;
```

**`example/46_gemm_add_multiply/common.hpp`**
```
template <ck::index_t... Is>
using S = ck::Sequence<Is...>;
using Row = ck::tensor_layout::gemm::RowMajor;
using Col = ck::tensor_layout::gemm::ColumnMajor;
```

**`example/46_gemm_add_multiply/gemm_add_multiply_dl_fp16.cpp`**
```
using ADataType   = F16;
using BDataType   = F16;
using AccDataType = F32;
using D0DataType  = F16;
```

**`example/46_gemm_add_multiply/gemm_add_multiply_xdl_fp16.cpp`**
```
using ADataType   = F16;
using BDataType   = F16;
using AccDataType = F32;
using D0DataType  = F16;
```

**`example/46_gemm_add_multiply/run_gemm_add_multiply_example.inc`**
```
bool run_gemm_add_multiply(const ProblemSize& problem_size, const ExecutionConfig& config)
using namespace ck::literals;
auto& [M, N, K, StrideA, StrideB, StrideD0, StrideD1, StrideE] = problem_size;
auto f_host_tensor_descriptor =
```
