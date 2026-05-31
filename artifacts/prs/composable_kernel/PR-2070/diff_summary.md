# Diff summary

- **files changed:** 11
- **lines:** +423 / -135
- **kernel-ish files:** 10

## Files (by churn)

- `test/gemm_universal/test_gemm_universal_ut_cases_fp16.inc`  (+113/-0)
- `test/gemm_universal/test_gemm_universal_ut_cases_fp8.inc`  (+113/-0)
- `test/gemm_universal/test_gemm_universal_xdl_fp16.cpp`  (+82/-0)
- `test/gemm_universal/test_gemm_universal_xdl_fp8.cpp`  (+71/-0)
- `test/gemm_universal/test_gemm_universal_ut_cases_bf16.inc`  (+14/-46)
- `test/gemm_universal/test_gemm_universal_xdl_bf16.cpp`  (+11/-23)
- `test/gemm_universal_streamk/test_gemm_universal_streamk_ut_cases_fp16.inc`  (+0/-28)
- `test/gemm_universal_streamk/test_gemm_universal_streamk_ut_cases_fp8.inc`  (+0/-28)
- `test/gemm_universal/CMakeLists.txt`  (+13/-2)
- `test/gemm_universal_streamk/test_gemm_universal_streamk_util.hpp`  (+5/-7)
- `profiler/include/profiler/profile_gemm_universal_streamk_impl.hpp`  (+1/-1)

## Key added lines (kernel files)

**`profiler/include/profiler/profile_gemm_universal_streamk_impl.hpp`**
```
if(Grid_size == -1)
```

**`test/gemm_universal/test_gemm_universal_ut_cases_bf16.inc`**
```
TYPED_TEST(TestGemmUniversal_BF16_MK_KN, SmallM)
TYPED_TEST(TestGemmUniversal_BF16_MK_NK, SmallM)
TYPED_TEST(TestGemmUniversal_BF16_KM_KN, SmallM)
TYPED_TEST(TestGemmUniversal_BF16_KM_NK, SmallM)
```

**`test/gemm_universal/test_gemm_universal_ut_cases_fp16.inc`**
```
TYPED_TEST(TestGemmUniversal_FP16_MK_KN, SmallM)
std::vector<int> Ms{1, 2, 3, 4, 5, 6};
constexpr int N = 512;
constexpr int K = 320;
```

**`test/gemm_universal/test_gemm_universal_ut_cases_fp8.inc`**
```
TYPED_TEST(TestGemmUniversal_FP8_MK_KN, SmallM)
std::vector<int> Ms{1, 2, 3, 4, 5, 6};
constexpr int N = 512;
constexpr int K = 320;
```

**`test/gemm_universal/test_gemm_universal_xdl_bf16.cpp`**
```
class TestGemmUniversal_BF16_MK_KN
class TestGemmUniversal_BF16_MK_NK
class TestGemmUniversal_BF16_KM_KN
class TestGemmUniversal_BF16_KM_NK
```
