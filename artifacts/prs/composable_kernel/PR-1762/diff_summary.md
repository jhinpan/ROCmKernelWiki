# Diff summary

- **files changed:** 37
- **lines:** +1583 / -350
- **kernel-ish files:** 31

## Files (by churn)

- `example/01_gemm/gemm_xdl_fp16_pk_i4_v3.cpp`  (+303/-0)
- `example/01_gemm/gemm_xdl_bf16_pk_i4_v3.cpp`  (+253/-0)
- `include/ck/tensor_operation/gpu/element/unary_element_wise_operation.hpp`  (+189/-0)
- `include/ck/tensor_operation/gpu/grid/gridwise_gemm_xdl_cshuffle_v3.hpp`  (+77/-27)
- `profiler/include/profiler/profile_gemm_universal_impl.hpp`  (+97/-6)
- `library/src/tensor_operation_instance/gpu/gemm_universal/device_gemm_xdl_universal_bf16_i4_bf16/device_gemm_xdl_universal_bf16_i4_bf16_mk_nk_mn.hpp`  (+87/-0)
- `library/src/tensor_operation_instance/gpu/gemm_universal/device_gemm_xdl_universal_f16_i4_f16/device_gemm_xdl_universal_f16_i4_f16_mk_nk_mn.hpp`  (+86/-0)
- `example/01_gemm/common.hpp`  (+82/-0)
- `example/01_gemm/run_gemm_example.inc`  (+0/-82)
- `example/01_gemm/run_gemm_example_streamk_v2.inc`  (+0/-82)
- `example/01_gemm/run_gemm_example_v2.inc`  (+0/-82)
- `include/ck/tensor_operation/gpu/thread/threadwise_tensor_slice_transfer_v3r1.hpp`  (+52/-22)
- `include/ck/library/utility/host_tensor.hpp`  (+55/-10)
- `include/ck/tensor_operation/gpu/thread/threadwise_tensor_slice_transfer.hpp`  (+45/-5)
- `include/ck/utility/data_type.hpp`  (+35/-0)

## Key added lines (kernel files)

**`example/01_gemm/common.hpp`**
```
template <typename DataType>
inline __host__ __device__ constexpr double get_rtol()
if constexpr(std::is_same_v<DataType, float>)
return 1e-3;
```

**`example/01_gemm/gemm_xdl_bf16_pk_i4_v3.cpp`**
```
using ADataType        = ck::bhalf_t;
using BDataType        = ck::pk_i4_t;
using AccDataType      = float;
using CShuffleDataType = ck::bhalf_t;
```

**`example/01_gemm/gemm_xdl_fp16_fp8_v3.cpp`**
```
using ADataType        = ck::half_t;
using BDataType        = ck::f8_t;
256, 8, 16,
S<32, 2, 1>,  S<1, 0, 2>,  S<1, 0, 2>,
```

**`example/01_gemm/gemm_xdl_fp16_pk_i4_v3.cpp`**
```
using ADataType        = ck::half_t;
using BDataType        = ck::pk_i4_t;
using AccDataType      = float;
using CShuffleDataType = ck::half_t;
```

**`example/01_gemm/gemm_xdl_fp16_v3.cpp`**
```
using BLayout = Col;
256, 8, 8,
S<32, 2, 1>,  S<1, 0, 2>,  S<1, 0, 2>,
S<32, 2, 1>,  S<1, 0, 2>,  S<1, 0, 2>,
```
