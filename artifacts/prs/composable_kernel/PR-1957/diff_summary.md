# Diff summary

- **files changed:** 13
- **lines:** +401 / -20
- **kernel-ish files:** 12

## Files (by churn)

- `include/ck_tile/ops/gemm/warp/warp_gemm_attribute_smfmac_impl.hpp`  (+114/-0)
- `include/ck_tile/ops/gemm/warp/warp_gemm_smfmac_impl.hpp`  (+110/-0)
- `include/ck_tile/ops/gemm/warp/warp_gemm_attribute_smfmac.hpp`  (+80/-0)
- `include/ck_tile/host/fill.hpp`  (+43/-0)
- `example/ck_tile/03_gemm/run_gemm_example.inc`  (+16/-8)
- `include/ck_tile/ops/gemm/warp/warp_gemm_dispatcher.hpp`  (+11/-4)
- `include/ck_tile/ops/gemm/warp/warp_gemm.hpp`  (+12/-1)
- `include/ck_tile/ops/gemm/pipeline/tile_gemm_traits.hpp`  (+5/-3)
- `include/ck_tile/ops/gemm/pipeline/gemm_universal_pipeline_ag_bg_cr_policy.hpp`  (+3/-1)
- `example/ck_tile/03_gemm/gemm_utils.hpp`  (+2/-1)
- `example/ck_tile/03_gemm/universal_gemm.cpp`  (+2/-1)
- `include/ck_tile/ops/gemm/pipeline/gemm_pipeline_problem.hpp`  (+2/-1)
- `CHANGELOG.md`  (+1/-0)

## Key added lines (kernel files)

**`example/ck_tile/03_gemm/gemm_utils.hpp`**
```
static constexpr bool TransposeC            = false;
static constexpr bool UseStructuredSparsity = false;
```

**`example/ck_tile/03_gemm/run_gemm_example.inc`**
```
GemmConfig::TransposeC,
GemmConfig::UseStructuredSparsity>;
std::cout << "Run Gemm kernel with M=" << M << " N=" << N << " K=" << K
<< " StrideA=" << stride_A << " StrideB=" << stride_B << " StrideC=" << stride_C
```

**`example/ck_tile/03_gemm/universal_gemm.cpp`**
```
GemmConfig::TransposeC,
GemmConfig::UseStructuredSparsity>;
```

**`include/ck_tile/host/fill.hpp`**
```
template <typename T>
struct AdjustToStructuredSparsity
size_t start{0};
static constexpr int32_t masks[] = {0, 0, 1, 1,
```

**`include/ck_tile/ops/gemm/pipeline/gemm_pipeline_problem.hpp`**
```
static constexpr bool TransposeC            = Traits::TransposeC;
static constexpr bool UseStructuredSparsity = Traits::UseStructuredSparsity;
```
