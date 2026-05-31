# Diff summary

- **files changed:** 19 (diff was byte-capped; summary is partial)
- **lines:** +3589 / -1517
- **kernel-ish files:** 12

## Files (by churn)

- `library/src/blas3/Tensile/gemm.cpp`  (+291/-1296)
- `library/src/blas3/Tensile/gemm.hpp`  (+922/-0)
- `clients/include/testing_gemm_batched_ex.hpp`  (+708/-0)
- `library/include/rocblas-functions.h`  (+607/-30)
- `clients/include/testing_gemm_batched.hpp`  (+496/-0)
- `clients/gtest/gemm_batched_gtest.yaml`  (+330/-0)
- `library/src/blas3/Tensile/gemm.h`  (+0/-133)
- `clients/include/testing_gemm_strided_batched.hpp`  (+40/-24)
- `clients/include/near.hpp`  (+62/-0)
- `clients/gtest/gemm_gtest.cpp`  (+47/-4)
- `clients/gtest/gemm_strided_batched_gtest.yaml`  (+17/-23)
- `clients/include/rocblas.hpp`  (+35/-2)
- `clients/benchmarks/client.cpp`  (+14/-3)
- `clients/include/unit.hpp`  (+11/-0)
- `library/src/CMakeLists.txt`  (+4/-0)

## Key added lines (kernel files)

**`clients/benchmarks/client.cpp`**
```
if(!strcmp(arg.function, "gemm_ex"))
testing_gemm_ex<Ti, To, Tc>(arg);
else if(!strcmp(arg.function, "gemm_batched_ex"))
testing_gemm_batched_ex<Ti, To, Tc>(arg);
```

**`clients/gtest/gemm_gtest.cpp`**
```
GEMM_BATCHED,
GEMM_BATCHED_EX,
case GEMM_BATCHED:
return !strcmp(arg.function, "gemm_batched")
```

**`clients/include/near.hpp`**
```
do                                                                               \
{                                                                                \
for(size_t k = 0; k < batch_count; k++)                                      \
for(size_t j = 0; j < N; j++)                                            \
```

**`clients/include/rocblas.hpp`**
```
template <typename T>
rocblas_status (*rocblas_gemm_batched)(rocblas_handle    handle,
rocblas_operation transA,
rocblas_operation transB,
```

**`clients/include/testing_gemm_batched.hpp`**
```
template <typename T>
void testing_gemm_batched(const Arguments& arg)
rocblas_int M = arg.M;
rocblas_int N = arg.N;
```
