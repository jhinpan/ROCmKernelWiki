# Diff summary

- **files changed:** 9
- **lines:** +79 / -409
- **kernel-ish files:** 8

## Files (by churn)

- `test/gemm_universal/test_gemm_universal_ut_cases_fp8.inc`  (+0/-113)
- `test/gemm_universal/test_gemm_universal_ut_cases_fp16.inc`  (+0/-99)
- `test/gemm_universal/test_gemm_universal_xdl_fp16.cpp`  (+0/-82)
- `test/gemm_universal/test_gemm_universal_xdl_fp8.cpp`  (+0/-71)
- `test/gemm_universal/test_gemm_universal_ut_cases.inc`  (+46/-14)
- `test/gemm_universal/test_gemm_universal_xdl.cpp`  (+23/-11)
- `test/gemm_universal/CMakeLists.txt`  (+2/-13)
- `test/gemm_universal_streamk/test_gemm_universal_streamk_util.hpp`  (+7/-5)
- `profiler/include/profiler/profile_gemm_universal_streamk_impl.hpp`  (+1/-1)

## Key added lines (kernel files)

**`profiler/include/profiler/profile_gemm_universal_streamk_impl.hpp`**
```
if(Grid_size != -1)
```

**`test/gemm_universal/test_gemm_universal_ut_cases.inc`**
```
TYPED_TEST(TestGemmUniversal_MK_KN, SmallM)
TYPED_TEST(TestGemmUniversal_MK_NK, SmallM)
TYPED_TEST(TestGemmUniversal_KM_KN, SmallM)
TYPED_TEST(TestGemmUniversal_KM_NK, SmallM)
```

**`test/gemm_universal/test_gemm_universal_xdl.cpp`**
```
using F8   = ck::f8_t;
using F16  = ck::half_t;
class TestGemmUniversal_MK_KN
class TestGemmUniversal_MK_NK
```

**`test/gemm_universal_streamk/test_gemm_universal_streamk_util.hpp`**
```
grid_size_list   = {38, 114, 228}; // {38, 76, 114, 152, 190, 228, 266, 304, 342, 380};
streamk_sel_list = {0, 1, 2};      // 0: Data Parallel (DP) mode (Stream-K OFF), 1: 1-tile
for(auto grid_size : grid_size_list)
RunSingle(M, N, K, StrideA, StrideB, StrideC, streamk_sel, grid_size);
```
