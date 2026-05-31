# Diff summary

- **files changed:** 11
- **lines:** +174 / -158
- **kernel-ish files:** 11

## Files (by churn)

- `include/ck_tile/ops/gemm/pipeline/gemm_universal_pipeline_ag_bg_cr_policy.hpp`  (+47/-47)
- `include/ck_tile/core/algorithm/static_encoding_pattern.hpp`  (+43/-40)
- `include/ck_tile/ops/gemm_group_quant/pipeline/gemm_aquant_pipeline_ag_bg_cr_policy.hpp`  (+28/-29)
- `test/ck_tile/utility/print/test_print_static_encoding_pattern.cpp`  (+18/-6)
- `include/ck_tile/ops/epilogue/cshuffle_epilogue.hpp`  (+8/-7)
- `include/ck_tile/ops/gemm_group_quant/pipeline/gemm_bquant_pipeline_ag_bg_cr_policy.hpp`  (+7/-7)
- `include/ck_tile/ops/gemm_group_quant/pipeline/gemm_group_quant_utils.hpp`  (+7/-6)
- `include/ck_tile/ops/batched_transpose/pipeline/batched_transpose_common_policy.hpp`  (+6/-6)
- `include/ck_tile/ops/batched_transpose/pipeline/batched_transpose_policy.hpp`  (+6/-6)
- `include/ck_tile/ops/gemm_group_quant/pipeline/gemm_aquant_pipeline_ag_bg_cr_v3.hpp`  (+2/-2)
- `include/ck_tile/ops/gemm_group_quant/pipeline/gemm_bquant_pipeline_ag_bg_cr_v3.hpp`  (+2/-2)

## Key added lines (kernel files)

**`include/ck_tile/core/algorithm/static_encoding_pattern.hpp`**
```
struct tile_distribution_encoding_pattern
struct tile_distribution_encoding_pattern_2d : public tile_distribution_encoding_pattern
struct tile_distribution_encoding_pattern_2d<BlockSize,
YPerTile,
```

**`include/ck_tile/ops/batched_transpose/pipeline/batched_transpose_common_policy.hpp`**
```
using TileEncodingPattern = tile_distribution_encoding_pattern_2d<kBlockSize,
kSecondDimPerBlock,
kLeadDimPerBlock,
kVectorSize,
```

**`include/ck_tile/ops/batched_transpose/pipeline/batched_transpose_policy.hpp`**
```
using TileEncodingPattern = tile_distribution_encoding_pattern_2d<BlockSize,
MPerBlock,
NPerBlock,
VecLoadSize,
```

**`include/ck_tile/ops/epilogue/cshuffle_epilogue.hpp`**
```
tile_distribution_encoding_pattern_2d<kBlockSize,
MPerIterationShuffle,
NPerIterationShuffle,
GetVectorSizeC(),
```

**`include/ck_tile/ops/gemm/pipeline/gemm_universal_pipeline_ag_bg_cr_policy.hpp`**
```
using TileEncodingPattern     = tile_distribution_encoding_pattern_2d<BlockSize,
KPerBlock,
NPerBlock,
VecLoadSize,
```
