# Diff summary

- **files changed:** 30 (diff was byte-capped; summary is partial)
- **lines:** +3780 / -545
- **kernel-ish files:** 25

## Files (by churn)

- `example/ck_tile/01_fmha/fmha_fwd.cpp`  (+604/-164)
- `include/ck_tile/ops/fmha/kernel/fmha_fwd_appendkv_kernel.hpp`  (+679/-0)
- `include/ck_tile/ops/fmha/kernel/fmha_fwd_splitkv_kernel.hpp`  (+182/-177)
- `example/ck_tile/01_fmha/codegen/ops/fmha_fwd_appendkv.py`  (+355/-0)
- `example/ck_tile/01_fmha/fmha_fwd.hpp`  (+293/-30)
- `include/ck_tile/ops/fmha/pipeline/block_fmha_fwd_appendkv_pipeline_default_policy.hpp`  (+288/-0)
- `include/ck_tile/ops/fmha/block/page_block_navigator.hpp`  (+279/-0)
- `include/ck_tile/ops/fmha/pipeline/block_fmha_fwd_appendkv_pipeline.hpp`  (+277/-0)
- `example/ck_tile/01_fmha/codegen/ops/fmha_fwd_splitkv.py`  (+127/-61)
- `example/ck_tile/01_fmha/script/smoke_test_fwd.sh`  (+94/-41)
- `example/ck_tile/01_fmha/utils.hpp`  (+94/-21)
- `include/ck_tile/ops/fmha/block/block_rotary_embedding.hpp`  (+108/-0)
- `example/ck_tile/01_fmha/rotary.hpp`  (+84/-0)
- `include/ck_tile/host/reference/reference_batched_rotary_position_embedding.hpp`  (+73/-0)
- `include/ck_tile/core/tensor/tile_window.hpp`  (+50/-0)

## Key added lines (kernel files)

**`example/ck_tile/01_fmha/codegen/cpp_symbol_map.py`**
```
ROPE_MAP = {
"no" : "ck_tile::RotaryEmbeddingEnum::NONE",
"inter"  : "ck_tile::RotaryEmbeddingEnum::INTERLEAVED",
"half" : "ck_tile::RotaryEmbeddingEnum::HALF_ROTATED"
```

**`example/ck_tile/01_fmha/codegen/ops/fmha_fwd_appendkv.py`**
```
import copy
from dataclasses import dataclass
import fnmatch
import itertools
```

**`example/ck_tile/01_fmha/codegen/ops/fmha_fwd_splitkv.py`**
```
DTYPE_BITS = {
"fp32": 32,
"fp16": 16,
"bf16": 16,
```

**`example/ck_tile/01_fmha/fmha_fwd.cpp`**
```
.insert("s_k", "-1", "seqlen_k (including new key/value), -1 means equal to s")
.insert("s_knew",
"seqlen_k for new key/value, 0 means not to use this at all; "
"-1 to choose s_knew in [1, s] randomly.")
```

**`example/ck_tile/01_fmha/fmha_fwd.hpp`**
```
void* lse_ptr;
void* o_ptr;
const void* seqstart_q_ptr;
const void* seqstart_k_ptr;
```
