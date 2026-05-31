# Diff summary

- **files changed:** 43 (diff was byte-capped; summary is partial)
- **lines:** +2101 / -2052
- **kernel-ish files:** 39

## Files (by churn)

- `clients/include/blas_ex/testing_gemm_ex3.hpp`  (+338/-335)
- `library/src/blas2/rocblas_gemv_kernels.cpp`  (+306/-300)
- `library/src/blas2/rocblas_hemv_symv_kernels.cpp`  (+143/-139)
- `library/src/blas2/rocblas_syr_kernels.cpp`  (+132/-132)
- `library/src/blas1/rocblas_dot_kernels.cpp`  (+101/-100)
- `library/src/blas2/rocblas_sbmv_kernels.cpp`  (+92/-92)
- `library/src/blas1/rocblas_scal_kernels.cpp`  (+72/-72)
- `library/src/blas2/rocblas_tpmv_kernels.cpp`  (+51/-51)
- `library/src/blas2/rocblas_trmv_kernels.cpp`  (+49/-48)
- `library/src/blas2/rocblas_gbmv_kernels.cpp`  (+48/-48)
- `library/src/blas2/rocblas_hbmv_kernels.cpp`  (+44/-44)
- `library/src/blas2/rocblas_spmv_kernels.cpp`  (+44/-44)
- `library/src/blas2/rocblas_her2_kernels.cpp`  (+42/-42)
- `library/src/blas2/rocblas_hpmv_kernels.cpp`  (+40/-40)
- `library/src/blas2/rocblas_syr2_kernels.cpp`  (+40/-40)

## Key added lines (kernel files)

**`clients/gtest/rocblas_test.cpp`**
```
if(hipPeekAtLastError() != hipSuccess)
rocblas_cerr << "hipGetLastError at end of test: "
<< ::testing::UnitTest::GetInstance()->current_test_info()->name()
<< std::endl;
```

**`clients/include/blas_ex/testing_gemm_ex3.hpp`**
```
rocblas_status call_trusted_gemm_f8(rocblas_handle    handle,
rocblas_operation transA,
rocblas_operation transB,
rocblas_int       M,
```

**`clients/include/blas_ex/testing_trsm_batched_ex.hpp`**
```
int remainder = K - TRSM_BLOCK * blocks;
if(remainder)
CHECK_ROCBLAS_ERROR(
rocblas_trtri_strided_batched<T>(handle,
```

**`clients/include/blas_ex/testing_trsm_ex.hpp`**
```
int remainder = K - TRSM_BLOCK * blocks;
if(remainder)
CHECK_ROCBLAS_ERROR(rocblas_trtri_strided_batched<T>(handle,
remainder,
```

**`clients/include/blas_ex/testing_trsm_strided_batched_ex.hpp`**
```
int remainder = K - TRSM_BLOCK * blocks;
if(remainder)
CHECK_ROCBLAS_ERROR(rocblas_trtri_strided_batched<T>(
remainder,
```
