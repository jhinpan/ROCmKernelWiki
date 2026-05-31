# Diff summary

- **files changed:** 24 (diff was byte-capped; summary is partial)
- **lines:** +4288 / -170
- **kernel-ish files:** 23

## Files (by churn)

- `include/ck_tile/ops/fmha/kernel/fmha_batch_prefill_kernel.hpp`  (+1134/-0)
- `include/ck_tile/ops/fmha/pipeline/block_fmha_batch_prefill_pipeline_qr_ks_vs_async.hpp`  (+900/-0)
- `include/ck_tile/core/tensor/tile_scatter_gather.hpp`  (+731/-0)
- `example/ck_tile/01_fmha/codegen/ops/fmha_batch_prefill.py`  (+595/-0)
- `include/ck_tile/ops/fmha/block/variants.hpp`  (+274/-0)
- `example/ck_tile/01_fmha/fmha_fwd.hpp`  (+212/-0)
- `include/ck_tile/ops/fmha/pipeline/block_fmha_fwd_splitkv_pipeline_nwarp_sshuffle_qr_ks_vs.hpp`  (+82/-16)
- `include/ck_tile/core/tensor/load_tile.hpp`  (+6/-84)
- `include/ck_tile/ops/fmha/kernel/fmha_fwd_kernel.hpp`  (+79/-4)
- `include/ck_tile/ops/fmha/kernel/fmha_fwd_splitkv_kernel.hpp`  (+72/-5)
- `example/ck_tile/01_fmha/codegen/ops/fmha_fwd.py`  (+44/-28)
- `example/ck_tile/01_fmha/codegen/ops/fmha_fwd_splitkv.py`  (+35/-22)
- `include/ck_tile/core/numeric/math.hpp`  (+41/-0)
- `include/ck_tile/core/tensor/tensor_view.hpp`  (+21/-0)
- `include/ck_tile/ops/fmha/pipeline/block_fmha_fwd_splitkv_pipeline_qr_ks_vs.hpp`  (+10/-9)

## Key added lines (kernel files)

**`example/ck_tile/01_fmha/codegen/cpp_symbol_map.py`**
```
"qs" : "ck_tile::BlockFmhaPipelineQSKSVS",
"qs" : "ck_tile::BlockFmhaPipelineEnum::QSKSVS",
```

**`example/ck_tile/01_fmha/codegen/ops/fmha_batch_prefill.py`**
```
import copy
from dataclasses import dataclass
import fnmatch
import itertools
```

**`example/ck_tile/01_fmha/codegen/ops/fmha_fwd.py`**
```
{F_logits},
using fmha_variant_{F_idx} = ck_tile::ComposedAttention<{F_logits} * ck_tile::LOGITS_SOFT_CAP, CK_TILE_FMHA_FWD_FAST_EXP
fmha_variant_{F_idx},
{F_pipeline_enum}, {F_logits}, fmha_mask_{F_idx}, {F_bias}, {F_lse}, {F_dropout}, {F_squant}, {F_spad}, {F_skpad}, {F_dp
```

**`example/ck_tile/01_fmha/codegen/ops/fmha_fwd_splitkv.py`**
```
using fmha_variant_{F_idx} = ck_tile::ComposedAttention<{F_logits} * ck_tile::LOGITS_SOFT_CAP, CK_TILE_FMHA_FWD_FAST_EXP
{F_logits},
fmha_variant_{F_idx},
{F_pipeline_enum}, {F_logits}, fmha_mask_{F_idx}, {F_bias}, {F_lse}, {F_squant}, {F_pagedkv}, {F_spad}, {F_skpad}, {F_dp
```

**`example/ck_tile/01_fmha/fmha_fwd.cpp`**
```
.insert("logits_soft_cap", "0", "attention logits soft capping value.")
const float logits_soft_cap = arg_parser.get_float("logits_soft_cap");
traits.has_logits_soft_cap = 0.f < logits_soft_cap;
args.logits_soft_cap = logits_soft_cap;
```
