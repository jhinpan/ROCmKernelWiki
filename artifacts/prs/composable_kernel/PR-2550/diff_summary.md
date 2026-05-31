# Diff summary

- **files changed:** 11
- **lines:** +158 / -189
- **kernel-ish files:** 11

## Files (by churn)

- `example/ck_tile/01_fmha/codegen/ops/fmha_bwd.py`  (+65/-106)
- `include/ck_tile/ops/fmha/kernel/fmha_bwd_kernel.hpp`  (+20/-34)
- `include/ck_tile/ops/fmha/pipeline/block_fmha_bwd_dq_dk_dv_pipeline_selector.hpp`  (+30/-0)
- `example/ck_tile/01_fmha/fmha_bwd.hpp`  (+11/-17)
- `example/ck_tile/01_fmha/codegen/utils.py`  (+21/-0)
- `include/ck_tile/ops/fmha/pipeline/block_fmha_bwd_pipeline_enum.hpp`  (+0/-15)
- `include/ck_tile/core/tensor/null_tile_window.hpp`  (+5/-2)
- `include/ck_tile/ops/fmha/pipeline/block_fmha_bwd_dq_dk_dv_pipeline_kr_ktr_vr_iglp.hpp`  (+1/-6)
- `include/ck_tile/ops/fmha/pipeline/block_fmha_bwd_dq_dk_dv_pipeline_kr_ktr_vr.hpp`  (+1/-5)
- `include/ck_tile/ops/fmha/pipeline/block_fmha_bwd_pipeline_problem.hpp`  (+3/-3)
- `include/ck_tile/ops/fmha.hpp`  (+1/-1)

## Key added lines (kernel files)

**`example/ck_tile/01_fmha/codegen/ops/fmha_bwd.py`**
```
from collections import defaultdict
from codegen.utils import update_file
using fmha_bwd_trait_{F_idx} = ck_tile::TileFmhaTraits<false,  /* kPadSeqLenQ */
false,  /* kPadSeqLenK */
```

**`example/ck_tile/01_fmha/codegen/utils.py`**
```
import os.path as path
def update_file(file_path, content):
"""Update the file at file_path with the given content if it differs from the existing content.
It avoids unnecessary touching of the file which triggers rebuilds
```

**`example/ck_tile/01_fmha/fmha_bwd.hpp`**
```
static constexpr ck_tile::index_t HDim = HDim_;
using DataType                         = ck_tile::remove_cvref_t<DataType_>;
static constexpr bool kIsGroupMode     = kIsGroupMode_;
using FmhaMask                         = ck_tile::remove_cvref_t<FmhaMask_>;
```

**`include/ck_tile/core/tensor/null_tile_window.hpp`**
```
template <typename T>
constexpr bool is_null_tile_window_v = impl::is_null_tile_window<remove_cvref_t<T>>::value;
return is_null_tile_window_v<remove_cvref_t<T>>;
```

**`include/ck_tile/ops/fmha/kernel/fmha_bwd_kernel.hpp`**
```
("o" + _TS_(kBlockPerCu)) + (pn.empty() ? "_npad" : "_" + pn) +
sequence<false, kPadHeadDimQ>{});
sequence<false, kPadHeadDimQ>{});
sequence<false, kPadHeadDimV>{});
```
