# Diff summary

- **files changed:** 8
- **lines:** +128 / -138
- **kernel-ish files:** 7

## Files (by churn)

- `include/ck_tile/host/reference/reference_gemm.hpp`  (+69/-70)
- `test/ck_tile/gemm_block_scale/test_gemm_quant_abquant_padding.cpp`  (+39/-0)
- `example/ck_tile/38_block_scale_gemm/run_gemm_quant_example.inc`  (+0/-35)
- `include/ck_tile/ops/gemm_quant/kernel/gemm_quant_kernel.hpp`  (+0/-26)
- `test/ck_tile/gemm_block_scale/CMakeLists.txt`  (+10/-4)
- `test/ck_tile/gemm_block_scale/test_gemm_quant_base.hpp`  (+4/-3)
- `test/ck_tile/gemm_block_scale/test_gemm_quant_fixtures.hpp`  (+6/-0)
- `test/ck_tile/gemm_block_scale/test_gemm_quant_abquant_base.cpp`  (+0/-0)

## Key added lines (kernel files)

**`include/ck_tile/host/reference/reference_gemm.hpp`**
```
AccDataType v_acc = 0;
constexpr std::size_t kGroupK = BQuantGroupSize::kK;
auto load_a = [&](std::size_t k) -> AccDataType {
return (k & 1) ? fp32_val.hi : fp32_val.lo;
```

**`test/ck_tile/gemm_block_scale/test_gemm_quant_abquant_padding.cpp`**
```
using RowMajor    = ck_tile::tensor_layout::gemm::RowMajor;
using ColumnMajor = ck_tile::tensor_layout::gemm::ColumnMajor;
using FP8         = ck_tile::fp8_t;
using BF8         = ck_tile::bf8_t;
```

**`test/ck_tile/gemm_block_scale/test_gemm_quant_base.hpp`**
```
static constexpr bool kPadM = GemmConfig::kPadM;
static constexpr bool kPadN = GemmConfig::kPadN;
static constexpr bool kPadK = GemmConfig::kPadK;
```

**`test/ck_tile/gemm_block_scale/test_gemm_quant_fixtures.hpp`**
```
struct GemmConfigPadding : public GemmConfigBase
static constexpr bool kPadN = true;
static constexpr bool kPadK = true;
```
