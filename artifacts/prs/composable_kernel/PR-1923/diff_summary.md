# Diff summary

- **files changed:** 12
- **lines:** +90 / -48
- **kernel-ish files:** 12

## Files (by churn)

- `include/ck/utility/type_convert.hpp`  (+37/-7)
- `codegen/include/ck/host/device_batched_gemm_softmax_gemm/problem.hpp`  (+18/-17)
- `codegen/src/device_batched_gemm_softmax_gemm_operation_xdl_cshuffle.cpp`  (+9/-5)
- `include/ck/utility/amd_ck_fp8.hpp`  (+5/-4)
- `codegen/test/batched_gemm_softmax_gemm.cpp`  (+3/-5)
- `include/ck/utility/mxf6_utils.hpp`  (+4/-4)
- `include/ck/utility/mxfp_utils.hpp`  (+5/-1)
- `include/ck/utility/mxf4_utils.hpp`  (+3/-2)
- `include/ck/utility/data_type.hpp`  (+2/-2)
- `include/ck/tensor_operation/gpu/device/device_gemm_multiple_d.hpp`  (+2/-0)
- `include/ck/tensor_operation/gpu/device/impl/device_batched_gemm_softmax_gemm_xdl_cshuffle.hpp`  (+1/-1)
- `codegen/test/rtc/src/compile_kernel.cpp`  (+1/-0)

## Key added lines (kernel files)

**`codegen/include/ck/host/device_batched_gemm_softmax_gemm/problem.hpp`**
```
std::size_t M             = 0;
std::size_t N             = 0;
std::size_t K             = 0;
std::size_t O             = 0;
```

**`codegen/src/device_batched_gemm_softmax_gemm_operation_xdl_cshuffle.cpp`**
```
x.mask_out_upper_triangle = prob.MaskOutUpperTriangle;
std::vector<Problem> problems;
problems.push_back(prob);
prob.MaskOutUpperTriangle = true;
```

**`codegen/test/batched_gemm_softmax_gemm.cpp`**
```
check_all<half> check;
CHECK(report(solution, check(rtc::from_gpu(c))));
```

**`codegen/test/rtc/src/compile_kernel.cpp`**
```
options.flags += " -DCK_CODE_GEN_RTC";
```

**`include/ck/utility/amd_ck_fp8.hpp`**
```
typename ck::conditional_t<
typename ck::conditional_t<sizeof(T) == 4, unsigned int, unsigned long long>>
using T_bitwise = typename ck::conditional_t<
typename ck::conditional_t<sizeof(T) == 4, unsigned int, unsigned long long>>;
```
