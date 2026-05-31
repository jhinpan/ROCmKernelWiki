# Diff summary

- **files changed:** 17
- **lines:** +314 / -13
- **kernel-ish files:** 13

## Files (by churn)

- `test/ck_tile/gemm_streamk/test_gemm_streamk_types_bf8.hpp`  (+77/-0)
- `test/ck_tile/gemm_streamk/test_gemm_streamk_types_fp8.hpp`  (+77/-0)
- `test/ck_tile/gemm_streamk/CMakeLists.txt`  (+33/-8)
- `example/ck_tile/40_streamk_gemm/gemm_utils.hpp`  (+13/-1)
- `example/ck_tile/40_streamk_gemm/streamk_gemm_basic.cpp`  (+13/-1)
- `test/ck_tile/gemm_streamk/smoke_tests/bf8_ccr_compv3_128x128x32_2x2x1_32x32x16_NonPersistent.cpp`  (+11/-0)
- `test/ck_tile/gemm_streamk/smoke_tests/bf8_crr_compv3_128x128x32_2x2x1_32x32x16_NonPersistent.cpp`  (+11/-0)
- `test/ck_tile/gemm_streamk/smoke_tests/bf8_rcr_compv3_128x128x32_2x2x1_32x32x16_NonPersistent.cpp`  (+11/-0)
- `test/ck_tile/gemm_streamk/smoke_tests/bf8_rrr_compv3_128x128x32_2x2x1_32x32x16_NonPersistent.cpp`  (+11/-0)
- `test/ck_tile/gemm_streamk/smoke_tests/f8_ccr_compv3_128x128x32_2x2x1_32x32x16_NonPersistent.cpp`  (+11/-0)
- `test/ck_tile/gemm_streamk/smoke_tests/f8_crr_compv3_128x128x32_2x2x1_32x32x16_NonPersistent.cpp`  (+11/-0)
- `test/ck_tile/gemm_streamk/smoke_tests/f8_rcr_compv3_128x128x32_2x2x1_32x32x16_NonPersistent.cpp`  (+11/-0)
- `test/ck_tile/gemm_streamk/smoke_tests/f8_rrr_compv3_128x128x32_2x2x1_32x32x16_NonPersistent.cpp`  (+11/-0)
- `test/ck_tile/gemm_streamk/test_gemm_streamk_types.hpp`  (+5/-1)
- `example/ck_tile/40_streamk_gemm/CMakeLists.txt`  (+5/-0)

## Key added lines (kernel files)

**`example/ck_tile/40_streamk_gemm/gemm_utils.hpp`**
```
template <>
struct DataTypeTraits<ck_tile::fp8_t>
static constexpr const char* name = "fp8";
template <>
```

**`example/ck_tile/40_streamk_gemm/streamk_gemm_basic.cpp`**
```
using GemmPipeline = ck_tile::GemmPipelineAgBgCrCompV3<UniversalGemmProblem>;
else if(data_type == "fp8")
using TypeConfig = StreamKGemmTypeConfig<ck_tile::fp8_t, ck_tile::fp8_t, ck_tile::half_t>;
return run_gemm_example_prec_type<GemmConfig<ck_tile::fp8_t>, TypeConfig>(
```

**`test/ck_tile/gemm_streamk/smoke_tests/bf8_ccr_compv3_128x128x32_2x2x1_32x32x16_NonPersistent.cpp`**
```
DECLARE_STREAM_K_TEST(TEST_SUITE_NAME, TEST_SUITE_PARAMS);
```

**`test/ck_tile/gemm_streamk/smoke_tests/bf8_crr_compv3_128x128x32_2x2x1_32x32x16_NonPersistent.cpp`**
```
DECLARE_STREAM_K_TEST(TEST_SUITE_NAME, TEST_SUITE_PARAMS);
```

**`test/ck_tile/gemm_streamk/smoke_tests/bf8_rcr_compv3_128x128x32_2x2x1_32x32x16_NonPersistent.cpp`**
```
DECLARE_STREAM_K_TEST(TEST_SUITE_NAME, TEST_SUITE_PARAMS);
```
