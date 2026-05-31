# Diff summary

- **files changed:** 106
- **lines:** +1947 / -339
- **kernel-ish files:** 104

## Files (by churn)

- `test/ck_tile/gemm_streamk/test_gemm_streamk_util.hpp`  (+63/-232)
- `test/ck_tile/gemm_streamk/test_gemm_streamk.hpp`  (+269/-0)
- `test/ck_tile/gemm_streamk/test_gemm_streamk_cases.inc`  (+143/-87)
- `test/ck_tile/gemm_streamk/CMakeLists.txt`  (+115/-1)
- `test/ck_tile/gemm_streamk/test_gemm_streamk_types.hpp`  (+104/-6)
- `test/ck_tile/gemm_streamk/test_gemm_streamk_types_fp16.hpp`  (+77/-0)
- `test/ck_tile/gemm_streamk/test_gemm_streamk_types_bf16.hpp`  (+76/-0)
- `include/ck_tile/ops/gemm/kernel/streamk_gemm_kernel.hpp`  (+41/-5)
- `test/ck_tile/gemm_streamk/extended_tests/compv3/bf16_ccc_compv3_128x128x32_2x2x1_32x32x16_NonPersistent.cpp`  (+11/-0)
- `test/ck_tile/gemm_streamk/extended_tests/compv3/bf16_ccc_compv3_256x256x32_2x2x1_32x32x16_NonPersistent.cpp`  (+11/-0)
- `test/ck_tile/gemm_streamk/extended_tests/compv3/bf16_ccr_compv3_128x128x32_2x2x1_32x32x16_NonPersistent.cpp`  (+11/-0)
- `test/ck_tile/gemm_streamk/extended_tests/compv3/bf16_ccr_compv3_256x256x32_2x2x1_32x32x16_NonPersistent.cpp`  (+11/-0)
- `test/ck_tile/gemm_streamk/extended_tests/compv3/bf16_crc_compv3_128x128x32_2x2x1_32x32x16_NonPersistent.cpp`  (+11/-0)
- `test/ck_tile/gemm_streamk/extended_tests/compv3/bf16_crc_compv3_256x256x32_2x2x1_32x32x16_NonPersistent.cpp`  (+11/-0)
- `test/ck_tile/gemm_streamk/extended_tests/compv3/bf16_crr_compv3_128x128x32_2x2x1_32x32x16_NonPersistent.cpp`  (+11/-0)

## Key added lines (kernel files)

**`include/ck_tile/ops/gemm/kernel/streamk_gemm_kernel.hpp`**
```
index_t i_m         = static_cast<index_t>(spatial_idx[UniversalGemmKernel::I0] *
index_t i_n         = static_cast<index_t>(spatial_idx[UniversalGemmKernel::I1] *
auto [i_k_a, i_k_b] = GetKOffsets<ALayout, BLayout>(
static_cast<index_t>(iter_offset), kargs.stride_As[0], kargs.stride_Bs[0]);
```

**`test/ck_tile/gemm_streamk/extended_tests/compv3/bf16_ccc_compv3_128x128x32_2x2x1_32x32x16_NonPersistent.cpp`**
```
DECLARE_STREAM_K_TEST(TEST_SUITE_NAME, TEST_SUITE_PARAMS);
```

**`test/ck_tile/gemm_streamk/extended_tests/compv3/bf16_ccc_compv3_256x256x32_2x2x1_32x32x16_NonPersistent.cpp`**
```
DECLARE_STREAM_K_TEST(TEST_SUITE_NAME, TEST_SUITE_PARAMS);
```

**`test/ck_tile/gemm_streamk/extended_tests/compv3/bf16_ccr_compv3_128x128x32_2x2x1_32x32x16_NonPersistent.cpp`**
```
DECLARE_STREAM_K_TEST(TEST_SUITE_NAME, TEST_SUITE_PARAMS);
```

**`test/ck_tile/gemm_streamk/extended_tests/compv3/bf16_ccr_compv3_256x256x32_2x2x1_32x32x16_NonPersistent.cpp`**
```
DECLARE_STREAM_K_TEST(TEST_SUITE_NAME, TEST_SUITE_PARAMS);
```
