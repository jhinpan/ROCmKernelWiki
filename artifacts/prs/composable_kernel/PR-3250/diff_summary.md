# Diff summary

- **files changed:** 25
- **lines:** +195 / -940
- **kernel-ish files:** 21

## Files (by churn)

- `include/ck_tile/ops/fmha/block/block_masking.hpp`  (+16/-162)
- `include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_async.hpp`  (+23/-79)
- `include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs.hpp`  (+20/-79)
- `include/ck_tile/ops/fmha/pipeline/block_fmha_fwd_pagedkv_pipeline_qr_ks_vs.hpp`  (+23/-71)
- `include/ck_tile/ops/fmha/pipeline/block_fmha_fwd_splitkv_pipeline_qr_ks_vs.hpp`  (+19/-73)
- `include/ck_tile/ops/fmha/pipeline/block_fmha_fwd_splitkv_pipeline_nwarp_sshuffle_qr_ks_vs.hpp`  (+21/-65)
- `example/ck_tile/01_fmha/script/smoke_test_fwd_sink.sh`  (+0/-83)
- `example/ck_tile/01_fmha/codegen/ops/fmha_fwd.py`  (+28/-46)
- `example/ck_tile/01_fmha/script/correct_test_fwd_sink.sh`  (+0/-74)
- `example/ck_tile/01_fmha/codegen/ops/fmha_fwd_splitkv.py`  (+13/-29)
- `example/ck_tile/01_fmha/mask.hpp`  (+5/-37)
- `example/ck_tile/01_fmha/codegen/ops/fmha_pagedkv_prefill.py`  (+10/-23)
- `include/ck_tile/ops/fmha/block/variants.hpp`  (+0/-33)
- `example/ck_tile/01_fmha/fmha_fwd.hpp`  (+2/-20)
- `include/ck_tile/ops/fmha/kernel/fmha_fwd_kernel.hpp`  (+2/-17)

## Key added lines (kernel files)

**`example/ck_tile/01_fmha/codegen/ops/fmha_fwd.py`**
```
{F_skip}>;
{F_pipeline_enum}, {F_logits}, fmha_mask_{F_idx}, {F_bias}, {F_lse}, {F_dropout}, {F_squant}, {F_spad}, {F_skpad}, {F_dp
FMHA_FWD_API_INNER_DISPATCH = """{F_if}((t.is_group_mode == {F_mode}) && (t.is_v_rowmajor == {F_vlayout}) && (t.has_logi
using trait_ = fmha_fwd_traits_<{F_hdim}, {F_dtype}, {F_mode}, {F_bm0}, {F_bn0}, {F_bk0}, {F_bn1}, {F_bk1}, {F_bk0max}, 
```

**`example/ck_tile/01_fmha/codegen/ops/fmha_fwd_splitkv.py`**
```
{F_occupancy}>;
{F_pipeline_enum}, {F_logits}, fmha_mask_{F_idx}, {F_bias}, {F_lse}, {F_squant}, {F_pagedkv}, {F_spad}, {F_skpad}, {F_dp
((a.block_table_ptr != nullptr) == {F_pagedkv}) && ({F_scheck}) && ({F_skcheck}) && ({F_dcheck}) && ({F_dvcheck})) {{
using traits_ = fmha_fwd_splitkv_traits_<{F_hdim}, {F_dtype}, {F_mode}, {F_bm0}, {F_bn0}, {F_bk0}, {F_bn1}, {F_bk1}, {F_
```

**`example/ck_tile/01_fmha/codegen/ops/fmha_pagedkv_prefill.py`**
```
{F_skip}>;
{F_pipeline_enum}, {F_logits}, fmha_mask_{F_idx}, {F_bias}, {F_lse}, {F_pagedkv}, {F_squant}, {F_spad}, {F_skpad}, {F_dp
FMHA_FWD_API_INNER_DISPATCH = """{F_if}((t.is_group_mode == {F_mode}) && (t.is_v_rowmajor == {F_vlayout}) && (t.has_logi
using trait_ = fmha_fwd_pagedkv_traits_<{F_hdim}, {F_dtype}, {F_mode}, {F_bm0}, {F_bn0}, {F_bk0}, {F_bn1}, {F_bk1}, {F_b
```

**`example/ck_tile/01_fmha/fmha_fwd.hpp`**
```
bool kSkipMinSeqlenQ_ = false>
bool kSkipMinSeqlenQ_ = false>
```

**`example/ck_tile/01_fmha/fmha_fwd_runner.hpp`**
```
mask.left, mask.right, real_seqlen_q, real_seqlen_k));
```
