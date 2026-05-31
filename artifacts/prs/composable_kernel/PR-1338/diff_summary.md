# Diff summary

- **files changed:** 20 (diff was byte-capped; summary is partial)
- **lines:** +4294 / -1198
- **kernel-ish files:** 19

## Files (by churn)

- `example/ck_tile/01_fmha/generate.py`  (+33/-1184)
- `include/ck_tile/ops/fmha/kernel/fmha_fwd_splitkv_kernel.hpp`  (+913/-0)
- `example/ck_tile/01_fmha/codegen/ops/fmha_fwd_splitkv.py`  (+671/-0)
- `example/ck_tile/01_fmha/codegen/ops/fmha_bwd.py`  (+611/-0)
- `example/ck_tile/01_fmha/codegen/ops/fmha_fwd.py`  (+498/-0)
- `include/ck_tile/ops/fmha/kernel/fmha_fwd_splitkv_combine_kernel.hpp`  (+455/-0)
- `include/ck_tile/ops/fmha/pipeline/block_fmha_fwd_splitkv_combine_pipeline.hpp`  (+314/-0)
- `example/ck_tile/01_fmha/fmha_fwd.hpp`  (+215/-0)
- `include/ck_tile/ops/fmha/pipeline/block_fmha_fwd_splitkv_combine_pipeline_default_policy.hpp`  (+175/-0)
- `example/ck_tile/01_fmha/fmha_fwd.cpp`  (+160/-10)
- `example/ck_tile/01_fmha/codegen/cpp_symbol_map.py`  (+92/-0)
- `include/ck_tile/ops/fmha/kernel/fmha_fwd_splitkv_tile_partitioner.hpp`  (+53/-0)
- `include/ck_tile/ops/fmha/kernel/fmha_fwd_splitkv_combine_tile_partitioner.hpp`  (+49/-0)
- `include/ck_tile/ops/fmha/pipeline/block_fmha_fwd_splitkv_pipeline_qr_ks_vs.hpp`  (+19/-0)
- `include/ck_tile/ops/fmha/block/block_masking.hpp`  (+17/-0)

## Key added lines (kernel files)

**`example/ck_tile/01_fmha/codegen/cmake_config.py`**
```
GEN_DIR = ""    # in Cmake, have to generate files in same folder
```

**`example/ck_tile/01_fmha/codegen/cpp_symbol_map.py`**
```
DTYPE_MAP = {
"fp16": "ck_tile::fp16_t",
"bf16": "ck_tile::bf16_t",
"fp8" : "ck_tile::fp8_t"
```

**`example/ck_tile/01_fmha/codegen/ops/fmha_bwd.py`**
```
import copy
from dataclasses import dataclass
import fnmatch
import itertools
```

**`example/ck_tile/01_fmha/codegen/ops/fmha_fwd.py`**
```
import copy
from dataclasses import dataclass
import fnmatch
import itertools
```

**`example/ck_tile/01_fmha/codegen/ops/fmha_fwd_splitkv.py`**
```
import copy
from dataclasses import dataclass
import fnmatch
import itertools
```
