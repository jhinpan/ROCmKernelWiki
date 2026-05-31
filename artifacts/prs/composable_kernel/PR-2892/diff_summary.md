# Diff summary

- **files changed:** 25
- **lines:** +940 / -195
- **kernel-ish files:** 21

## Files (by churn)

- `include/ck_tile/ops/fmha/block/block_masking.hpp`  (+162/-16)
- `include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_async.hpp`  (+79/-23)
- `include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs.hpp`  (+79/-20)
- `include/ck_tile/ops/fmha/pipeline/block_fmha_fwd_pagedkv_pipeline_qr_ks_vs.hpp`  (+71/-23)
- `include/ck_tile/ops/fmha/pipeline/block_fmha_fwd_splitkv_pipeline_qr_ks_vs.hpp`  (+73/-19)
- `include/ck_tile/ops/fmha/pipeline/block_fmha_fwd_splitkv_pipeline_nwarp_sshuffle_qr_ks_vs.hpp`  (+65/-21)
- `example/ck_tile/01_fmha/script/smoke_test_fwd_sink.sh`  (+83/-0)
- `example/ck_tile/01_fmha/codegen/ops/fmha_fwd.py`  (+46/-28)
- `example/ck_tile/01_fmha/script/correct_test_fwd_sink.sh`  (+74/-0)
- `example/ck_tile/01_fmha/codegen/ops/fmha_fwd_splitkv.py`  (+29/-13)
- `example/ck_tile/01_fmha/mask.hpp`  (+37/-5)
- `example/ck_tile/01_fmha/codegen/ops/fmha_pagedkv_prefill.py`  (+23/-10)
- `include/ck_tile/ops/fmha/block/variants.hpp`  (+33/-0)
- `example/ck_tile/01_fmha/fmha_fwd.hpp`  (+20/-2)
- `include/ck_tile/ops/fmha/kernel/fmha_fwd_kernel.hpp`  (+17/-2)

## Key added lines (kernel files)

**`example/ck_tile/01_fmha/codegen/ops/fmha_fwd.py`**
```
{F_skip},
{F_sink}>;
{F_pipeline_enum}, {F_logits}, fmha_mask_{F_idx}, {F_bias}, {F_lse}, {F_dropout}, {F_squant}, {F_spad}, {F_skpad}, {F_dp
FMHA_FWD_API_INNER_DISPATCH = """{F_if}((t.is_group_mode == {F_mode}) && (t.is_v_rowmajor == {F_vlayout}) && (t.has_logi
```

**`example/ck_tile/01_fmha/codegen/ops/fmha_fwd_splitkv.py`**
```
{F_occupancy},
{F_sink}>;
{F_pipeline_enum}, {F_logits}, fmha_mask_{F_idx}, {F_bias}, {F_lse}, {F_squant}, {F_pagedkv}, {F_sink}, {F_spad}, {F_skp
((a.block_table_ptr != nullptr) == {F_pagedkv}) && (t.has_sink == {F_sink}) && ({F_scheck}) && ({F_skcheck}) && ({F_dche
```

**`example/ck_tile/01_fmha/codegen/ops/fmha_pagedkv_prefill.py`**
```
{F_skip},
{F_sink}>;
{F_pipeline_enum}, {F_logits}, fmha_mask_{F_idx}, {F_bias}, {F_lse}, {F_pagedkv}, {F_squant}, {F_spad}, {F_skpad}, {F_dp
FMHA_FWD_API_INNER_DISPATCH = """{F_if}((t.is_group_mode == {F_mode}) && (t.is_v_rowmajor == {F_vlayout}) && (t.has_logi
```

**`example/ck_tile/01_fmha/fmha_fwd.hpp`**
```
ck_tile::index_t sink_size;
ck_tile::index_t sink_size;
ck_tile::index_t sink_size;
args.sink_size,
```

**`example/ck_tile/01_fmha/fmha_fwd_runner.hpp`**
```
traits.has_sink            = mask.sink > 0 ? true : false;
args.sink_size         = mask.sink;
mask.left, mask.right, mask.sink, real_seqlen_q, real_seqlen_k));
mask.sink,
```
