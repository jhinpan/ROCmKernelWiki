# Diff summary

- **files changed:** 10
- **lines:** +423 / -619
- **kernel-ish files:** 9

## Files (by churn)

- `test/ck_tile/gemm_multi_d/test_gemm_multi_d_ut_cases.inc`  (+334/-0)
- `test/ck_tile/gemm_multi_d/test_gemm_multi_d_ut_cases_cshuffle.inc`  (+0/-211)
- `test/ck_tile/gemm_multi_d/test_gemm_multi_d_ut_cases_default2d.inc`  (+0/-211)
- `include/ck_tile/ops/epilogue/default_2d_epilogue.hpp`  (+30/-78)
- `test/ck_tile/gemm_multi_d/test_gemm_multi_d_util.hpp`  (+43/-46)
- `test/ck_tile/gemm_multi_d/test_gemm_multi_d_default2d.cpp`  (+0/-43)
- `test/ck_tile/gemm_multi_d/test_gemm_multi_d.cpp`  (+14/-15)
- `include/ck_tile/ops/gemm/kernel/gemm_multi_d_kernel.hpp`  (+0/-6)
- `test/ck_tile/gemm_multi_d/CMakeLists.txt`  (+2/-4)
- `tile_engine/ops/gemm/codegen_utils.py`  (+0/-5)

## Key added lines (kernel files)

**`include/ck_tile/ops/epilogue/default_2d_epilogue.hpp`**
```
template <typename ODramWindowTmp, typename OAccTile>
CK_TILE_DEVICE auto
operator()(ODramWindowTmp& o_dram_window_tmp, const OAccTile& o_acc_tile, void* = nullptr) const
if constexpr(UseRawStore && (kPadM || kPadN))
```

**`test/ck_tile/gemm_multi_d/test_gemm_multi_d.cpp`**
```
std::tuple<    Row,     Col,     Row,     Row,      Row,      F16,       F16,          BF16,       BF16,       F32,     
std::tuple<    Row,     Col,     Row,     Row,      Row,      F16,       F16,          F32,        F32,        F32,     
std::tuple<    Row,     Col,     Row,     Row,      Row,      F16,       F16,          F32,        F32,        F32,     
std::tuple<    Row,     Col,     Row,     Row,      Row,      F8,        F8,           BF16,       BF16,       F32,     
```

**`test/ck_tile/gemm_multi_d/test_gemm_multi_d_ut_cases.inc`**
```
TYPED_TEST(TestCkTileGemmMultiD, TestCkTileGemmMultiDAddKBatch1_256x512x256)
constexpr int M      = 256;
constexpr int N      = 512;
constexpr int K      = 256;
```

**`test/ck_tile/gemm_multi_d/test_gemm_multi_d_util.hpp`**
```
using ALayout         = std::tuple_element_t<0, Tuple>;
using BLayout         = std::tuple_element_t<1, Tuple>;
using D0Layout        = std::tuple_element_t<2, Tuple>;
using D1Layout        = std::tuple_element_t<3, Tuple>;
```
