# Diff summary

- **files changed:** 17
- **lines:** +731 / -114
- **kernel-ish files:** 16

## Files (by churn)

- `include/ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_comp_v5.hpp`  (+379/-0)
- `include/ck_tile/ops/gemm/block/block_gemm_areg_breg_creg_v1.hpp`  (+120/-48)
- `include/ck_tile/core/algorithm/static_encoding_pattern.hpp`  (+66/-26)
- `include/ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_comp_v5_default_policy.hpp`  (+63/-0)
- `include/ck_tile/ops/gemm/pipeline/gemm_universal_pipeline_ag_bg_cr_policy.hpp`  (+32/-22)
- `example/ck_tile/03_gemm/gemm_utils.hpp`  (+32/-3)
- `include/ck_tile/ops/gemm/kernel/gemm_kernel.hpp`  (+12/-5)
- `include/ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_base.hpp`  (+6/-4)
- `example/ck_tile/03_gemm/universal_gemm.cpp`  (+5/-3)
- `include/ck_tile/ops/epilogue/cshuffle_epilogue.hpp`  (+5/-2)
- `include/ck_tile/ops/gemm/pipeline/tile_gemm_traits.hpp`  (+3/-1)
- `include/ck_tile/ops/gemm.hpp`  (+2/-0)
- `include/ck_tile/ops/gemm/pipeline/gemm_pipeline_problem.hpp`  (+2/-0)
- `CHANGELOG.md`  (+1/-0)
- `include/ck_tile/ops/gemm/pipeline/gemm_pipeline_ag_bg_cr_comp_v3.hpp`  (+1/-0)

## Key added lines (kernel files)

**`example/ck_tile/03_gemm/gemm_utils.hpp`**
```
static constexpr bool DoubleSmemBuffer          = false;
static constexpr ck_tile::index_t NumWaveGroups = 1;
static constexpr bool DoubleSmemBuffer          = false;
static constexpr ck_tile::index_t NumWaveGroups = 1;
```

**`example/ck_tile/03_gemm/universal_gemm.cpp`**
```
Persistent,
GemmConfig::NumWaveGroups>;
memory_operation,
GemmConfig::NumWaveGroups>>;
```

**`include/ck_tile/core/algorithm/static_encoding_pattern.hpp`**
```
tile_distribution_pattern DistributionPattern,
index_t NumWaveGroups = 1>
template <index_t BlockSize,
index_t YPerTile,
```

**`include/ck_tile/ops/epilogue/cshuffle_epilogue.hpp`**
```
memory_operation_enum MemoryOperation_,
index_t kNumWaveGroups_ = 1>
static constexpr index_t kNumWaveGroups                = kNumWaveGroups_;
tile_distribution_pattern::thread_raked,
```

**`include/ck_tile/ops/gemm/block/block_gemm_areg_breg_creg_v1.hpp`**
```
static constexpr index_t MWarp            = Traits::MWarp;
static constexpr index_t NWarp            = Traits::NWarp;
static constexpr bool UseDefaultScheduler = (Problem::NumWaveGroups != 1);
if constexpr(UseDefaultScheduler)
```
