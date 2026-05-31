# Diff summary

- **files changed:** 16
- **lines:** +387 / -141
- **kernel-ish files:** 15

## Files (by churn)

- `include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs_async.hpp`  (+56/-31)
- `include/ck_tile/ops/fmha/pipeline/block_fmha_fwd_splitkv_pipeline_qr_ks_vs.hpp`  (+53/-15)
- `include/ck_tile/ops/fmha/pipeline/block_fmha_pipeline_qr_ks_vs.hpp`  (+52/-14)
- `example/ck_tile/01_fmha/codegen/ops/fmha_fwd.py`  (+31/-22)
- `include/ck_tile/ops/fmha/kernel/fmha_fwd_kernel.hpp`  (+50/-3)
- `example/ck_tile/01_fmha/codegen/ops/fmha_fwd_splitkv.py`  (+27/-18)
- `include/ck_tile/ops/fmha/kernel/fmha_fwd_splitkv_kernel.hpp`  (+42/-3)
- `example/ck_tile/15_fused_moe/instances/fused_moe_api.cpp`  (+15/-14)
- `include/ck/utility/dynamic_buffer.hpp`  (+12/-9)
- `example/ck_tile/01_fmha/fmha_fwd.cpp`  (+17/-0)
- `example/ck_tile/01_fmha/fmha_fwd.hpp`  (+14/-0)
- `include/ck_tile/core/numeric/math.hpp`  (+6/-5)
- `include/ck/tensor_operation/gpu/grid/gridwise_moe_gemm.hpp`  (+4/-5)
- `example/ck_tile/01_fmha/CMakeLists.txt`  (+2/-2)
- `include/ck_tile/ops/fmha/pipeline/tile_fmha_traits.hpp`  (+4/-0)

## Key added lines (kernel files)

**`example/ck_tile/01_fmha/codegen/ops/fmha_fwd.py`**
```
{F_logits},
{F_pipeline_enum}, {F_logits}, fmha_mask_{F_idx}, {F_bias}, {F_lse}, {F_dropout}, {F_squant}, {F_spad}, {F_skpad}, {F_dp
FMHA_FWD_API_INNER_DISPATCH="""            {F_if}((t.is_group_mode == {F_mode}) && (t.is_v_rowmajor == {F_vlayout}) && (
using trait_ = fmha_fwd_traits_<{F_hdim}, {F_dtype}, {F_mode}, {F_bm0}, {F_bn0}, {F_bk0}, {F_bn1}, {F_bk1}, {F_bk0max}, 
```

**`example/ck_tile/01_fmha/codegen/ops/fmha_fwd_splitkv.py`**
```
{F_logits},
{F_pipeline_enum}, {F_logits}, fmha_mask_{F_idx}, {F_bias}, {F_lse}, {F_squant}, {F_pagedkv}, {F_spad}, {F_skpad}, {F_dp
FMHA_FWD_SPLITKV_API_INNER_DISPATCH="""            {F_if}((t.is_group_mode == {F_mode}) && (t.is_v_rowmajor == {F_vlayou
using traits_ = fmha_fwd_splitkv_traits_<{F_hdim}, {F_dtype}, {F_mode}, {F_bm0}, {F_bn0}, {F_bk0}, {F_bn1}, {F_bk1}, {F_
```

**`example/ck_tile/01_fmha/fmha_fwd.cpp`**
```
.insert("logits_soft_cap", "0", "attention logits soft capping value.")
const float logits_soft_cap = arg_parser.get_float("logits_soft_cap");
traits.has_logits_soft_cap = 0.f < logits_soft_cap;
args.logits_soft_cap = logits_soft_cap;
```

**`example/ck_tile/01_fmha/fmha_fwd.hpp`**
```
float logits_soft_cap;
float logits_soft_cap;
args.logits_soft_cap,
args.logits_soft_cap,
```

**`example/ck_tile/15_fused_moe/instances/fused_moe_api.cpp`**
```
a.topk_ids_ptr,          // const void* p_topk_ids;
a.topk_weight_ptr,       // const void* p_weights;
a.local_expert_mask_ptr, // const void* p_local_expert_mask;
a.sorted_token_ids_ptr,  // void* p_sorted_token_ids;
```
