# Diff summary

- **files changed:** 10
- **lines:** +619 / -423
- **kernel-ish files:** 9

## Files (by churn)

- `test/ck_tile/gemm_multi_d/test_gemm_multi_d_ut_cases.inc`  (+0/-334)
- `test/ck_tile/gemm_multi_d/test_gemm_multi_d_ut_cases_cshuffle.inc`  (+211/-0)
- `test/ck_tile/gemm_multi_d/test_gemm_multi_d_ut_cases_default2d.inc`  (+211/-0)
- `include/ck_tile/ops/epilogue/default_2d_epilogue.hpp`  (+78/-30)
- `test/ck_tile/gemm_multi_d/test_gemm_multi_d_util.hpp`  (+46/-43)
- `test/ck_tile/gemm_multi_d/test_gemm_multi_d_default2d.cpp`  (+43/-0)
- `test/ck_tile/gemm_multi_d/test_gemm_multi_d_cshuffle.cpp`  (+15/-14)
- `include/ck_tile/ops/gemm/kernel/gemm_multi_d_kernel.hpp`  (+6/-0)
- `test/ck_tile/gemm_multi_d/CMakeLists.txt`  (+4/-2)
- `tile_engine/ops/gemm/codegen_utils.py`  (+5/-0)

## Key added lines (kernel files)

**`include/ck_tile/ops/epilogue/default_2d_epilogue.hpp`**
```
typename DsDataType_,
typename DsLayout_,
typename CDElementwise_,
index_t kM_,
```

**`include/ck_tile/ops/gemm/kernel/gemm_multi_d_kernel.hpp`**
```
if(kargs.k_batch > 1)
return false;
```

**`test/ck_tile/gemm_multi_d/test_gemm_multi_d_cshuffle.cpp`**
```
std::tuple<    Row,     Col,     Row,     Row,      Row,      F16,       F16,          BF16,       BF16,       F32,     
std::tuple<    Row,     Col,     Row,     Row,      Row,      F16,       F16,          F32,        F32,        F32,     
std::tuple<    Row,     Col,     Row,     Row,      Row,      F16,       F16,          F32,        F32,        F32,     
std::tuple<    Row,     Col,     Row,     Row,      Row,      F8,        F8,           BF16,       BF16,       F32,     
```

**`test/ck_tile/gemm_multi_d/test_gemm_multi_d_default2d.cpp`**
```
using F16  = ck_tile::half_t;
using BF16 = ck_tile::bf16_t;
using F32  = float;
using F8   = ck_tile::fp8_t;
```

**`test/ck_tile/gemm_multi_d/test_gemm_multi_d_ut_cases_cshuffle.inc`**
```
TYPED_TEST(TestCkTileGemmMultiD, TestCkTileGemmMultiDKBatch1CShuffle_256x512x256)
constexpr int M      = 256;
constexpr int N      = 512;
constexpr int K      = 256;
```
