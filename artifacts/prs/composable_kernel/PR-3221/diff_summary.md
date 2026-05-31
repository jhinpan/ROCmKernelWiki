# Diff summary

- **files changed:** 11
- **lines:** +37 / -16
- **kernel-ish files:** 11

## Files (by churn)

- `include/ck_tile/ops/fmha/block/block_dropout.hpp`  (+8/-14)
- `include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qx_ks_vs_custom_policy.hpp`  (+13/-0)
- `include/ck_tile/ops/fmha/kernel/fmha_batch_prefill_kernel.hpp`  (+1/-1)
- `include/ck_tile/ops/fmha/kernel/fmha_fwd_kernel.hpp`  (+1/-1)
- `include/ck_tile/ops/fmha/pipeline/block_fmha_fwd_pagedkv_pipeline_qr_ks_vs.hpp`  (+2/-0)
- `include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs.hpp`  (+2/-0)
- `include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_async.hpp`  (+2/-0)
- `include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_async_trload.hpp`  (+2/-0)
- `include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_fp8.hpp`  (+2/-0)
- `include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_whole_k_prefetch.hpp`  (+2/-0)
- `include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qs_ks_vs.hpp`  (+2/-0)

## Key added lines (kernel files)

**`include/ck_tile/ops/fmha/block/block_dropout.hpp`**
```
if(is_store_randval)
const auto randval_store = cast_tile<RandValOutputDataType>(randval);
store_tile(randval_dram_window, randval_store);
move_tile_window(randval_dram_window, {0, kNPerStep});
```

**`include/ck_tile/ops/fmha/kernel/fmha_batch_prefill_kernel.hpp`**
```
number<FmhaPipeline::kAlignmentRandVal>{},
```

**`include/ck_tile/ops/fmha/kernel/fmha_fwd_kernel.hpp`**
```
number<FmhaPipeline::kAlignmentRandVal>{},
```

**`include/ck_tile/ops/fmha/pipeline/block_fmha_fwd_pagedkv_pipeline_qr_ks_vs.hpp`**
```
static constexpr index_t kAlignmentRandVal =
kPadSeqLenK ? 1 : Policy::template GetAlignmentRandVal<Problem>();
```

**`include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs.hpp`**
```
static constexpr index_t kAlignmentRandVal =
kPadSeqLenK ? 1 : Policy::template GetAlignmentRandVal<Problem>();
```
