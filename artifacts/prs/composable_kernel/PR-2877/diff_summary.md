# Diff summary

- **files changed:** 17
- **lines:** +1130 / -54
- **kernel-ish files:** 16

## Files (by churn)

- `include/ck_tile/ops/gemm_quant/pipeline/gemm_wp_bquant_pipeline_ag_bg_cr_v2.hpp`  (+471/-0)
- `include/ck_tile/ops/gemm_quant/block/block_universal_gemm_ar_flatbr_bquant_cr.hpp`  (+191/-0)
- `include/ck_tile/ops/gemm_quant/kernel/gemm_quant_kernel.hpp`  (+170/-19)
- `example/ck_tile/38_block_scale_gemm/gemm_quant_basic.cpp`  (+70/-10)
- `test/ck_tile/gemm_block_scale/test_gemm_quant_fixtures.hpp`  (+63/-7)
- `include/ck_tile/ops/gemm_quant/pipeline/gemm_wp_bquant_pipeline_ag_bg_cr_base_policy.hpp`  (+60/-0)
- `example/ck_tile/38_block_scale_gemm/run_gemm_quant_example.inc`  (+31/-4)
- `test/ck_tile/gemm_block_scale/test_gemm_quant_base.hpp`  (+25/-6)
- `example/ck_tile/38_block_scale_gemm/gemm_utils.hpp`  (+22/-2)
- `test/ck_tile/gemm_block_scale/test_gemm_quant_typed.cpp`  (+9/-0)
- `include/ck_tile/ops/gemm_quant/block/block_universal_gemm_as_aquant_bs_cr.hpp`  (+3/-3)
- `include/ck_tile/ops/gemm_quant/pipeline/gemm_quant_pipeline_problem.hpp`  (+3/-3)
- `test/ck_tile/gemm_block_scale/test_gemm_quant_ut_cases.inc`  (+5/-0)
- `include/ck_tile/ops/gemm_quant.hpp`  (+3/-0)
- `include/ck_tile/ops/gemm_quant/pipeline/tile_gemm_quant_traits.hpp`  (+2/-0)

## Key added lines (kernel files)

**`example/ck_tile/38_block_scale_gemm/gemm_quant_basic.cpp`**
```
GemmConfig::PreshuffleB,
QuantMode,
ALayout, // for AQLayout
BLayout, // for BQLayout
```

**`example/ck_tile/38_block_scale_gemm/gemm_utils.hpp`**
```
static constexpr bool PreshuffleB      = false;
template <typename PrecType>
struct GemmConfigPreshuffleB_Bquant_decode : public GemmConfigBase
static constexpr ck_tile::index_t M_Tile = 16;
```

**`example/ck_tile/38_block_scale_gemm/run_gemm_quant_example.inc`**
```
template <typename GemmConfig, typename T>
auto shuffle_b(const ck_tile::HostTensor<T>& t)
assert(t.get_lengths().size() == 2);
int n_                = t.get_lengths()[1];
```

**`include/ck_tile/ops/gemm/warp/warp_gemm_attribute_mfma.hpp`**
```
static constexpr index_t kCMLane     = Impl::kCMLane;
```

**`include/ck_tile/ops/gemm_quant/block/block_universal_gemm_ar_flatbr_bquant_cr.hpp`**
```
namespace ck_tile {
template <typename Problem_, typename BlockPolicy_>
struct BlockGemmWeightPreshuffleBQuantARegBRegCReg
using Problem         = remove_cvref_t<Problem_>;
```
