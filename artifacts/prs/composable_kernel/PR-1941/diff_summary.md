# Diff summary

- **files changed:** 15
- **lines:** +1418 / -145
- **kernel-ish files:** 15

## Files (by churn)

- `include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_whole_k_prefetch.hpp`  (+929/-0)
- `include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_whole_k_prefetch_default_policy.hpp`  (+379/-0)
- `include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qx_ks_vs_custom_policy.hpp`  (+40/-76)
- `include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_async.hpp`  (+18/-35)
- `include/ck_tile/ops/fmha/pipeline/block_fmha_fwd_splitkv_pipeline_nwarp_sshuffle_qr_ks_vs.hpp`  (+11/-5)
- `include/ck_tile/ops/fmha/pipeline/block_fmha_fwd_splitkv_pipeline_qr_ks_vs.hpp`  (+10/-5)
- `include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs.hpp`  (+10/-5)
- `include/ck_tile/ops/fmha/kernel/fmha_fwd_kernel.hpp`  (+8/-3)
- `include/ck_tile/ops/fmha/pipeline/block_fmha_fwd_splitkv_pipeline_nwarp_sshuffle_qr_ks_vs_default_policy.hpp`  (+4/-6)
- `include/ck_tile/ops/fmha.hpp`  (+3/-1)
- `include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_default_policy.hpp`  (+1/-3)
- `include/ck_tile/ops/fmha/pipeline/block_fmha_fwd_splitkv_pipeline_qr_ks_vs_default_policy.hpp`  (+1/-2)
- `include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_async_default_policy.hpp`  (+1/-2)
- `include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qs_ks_vs_default_policy.hpp`  (+1/-2)
- `include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qs_ks_vs.hpp`  (+2/-0)

## Key added lines (kernel files)

**`include/ck_tile/ops/fmha/kernel/fmha_fwd_kernel.hpp`**
```
static constexpr bool kUseAsyncCopy = FmhaPipeline::Policy::AsyncCopy;
constexpr bool kPadSeqLenK_ = kUseAsyncCopy ? kPadSeqLenK : false;
sequence<kPadSeqLenK_, kPadHeadDimQ>{});
constexpr bool kPadSeqLenK_ = kUseAsyncCopy ? kPadSeqLenK : false;
```

**`include/ck_tile/ops/fmha/pipeline/block_fmha_fwd_splitkv_pipeline_nwarp_sshuffle_qr_ks_vs.hpp`**
```
static_assert(kSubQKHeaddim <= 256, "hdim bigger than 256 is not suitable for this pipeline!");
return 1;
auto q_lds_window_for_load =
make_tile_window(q_lds,
```

**`include/ck_tile/ops/fmha/pipeline/block_fmha_fwd_splitkv_pipeline_nwarp_sshuffle_qr_ks_vs_default_policy.hpp`**
```
template <typename Problem>
return BasePolicy::template MakeQRegTileDistribution<Problem>();
```

**`include/ck_tile/ops/fmha/pipeline/block_fmha_fwd_splitkv_pipeline_qr_ks_vs.hpp`**
```
static_assert(kSubQKHeaddim <= 256, "hdim bigger than 256 is not suitable for this pipeline!");
return 1;
auto q_dram_window = make_tile_window(q_dram_block_window_tmp.get_bottom_tensor_view(),
q_dram_block_window_tmp.get_window_lengths(),
```

**`include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs.hpp`**
```
static_assert(kSubQKHeaddim <= 256, "hdim bigger than 256 is not suitable for this pipeline!");
return 1;
auto q_dram_window = make_tile_window(q_dram_block_window_tmp.get_bottom_tensor_view(),
q_dram_block_window_tmp.get_window_lengths(),
```
