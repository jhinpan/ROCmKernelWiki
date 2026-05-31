# Diff summary

- **files changed:** 22
- **lines:** +6058 / -0
- **kernel-ish files:** 20

## Files (by churn)

- `example/ck_tile/50_sparse_attn/codegen/ops/fmha_fwd_jenga.py`  (+867/-0)
- `example/ck_tile/50_sparse_attn/codegen/ops/fmha_fwd_vsa.py`  (+867/-0)
- `include/ck_tile/ops/sparse_attn/pipeline/block_fmha_pipeline_qr_ks_vs_async_jenga.hpp`  (+595/-0)
- `include/ck_tile/ops/sparse_attn/pipeline/block_fmha_pipeline_qr_ks_vs_async_vsa.hpp`  (+579/-0)
- `example/ck_tile/50_sparse_attn/test_vsa_sparse_attn.cpp`  (+486/-0)
- `include/ck_tile/ops/sparse_attn/kernel/fmha_fwd_jenga_kernel.hpp`  (+446/-0)
- `include/ck_tile/ops/sparse_attn/kernel/fmha_fwd_vsa_kernel.hpp`  (+438/-0)
- `example/ck_tile/50_sparse_attn/test_jenga_sparse_attn.cpp`  (+423/-0)
- `example/ck_tile/50_sparse_attn/fmha_fwd_trek.hpp`  (+328/-0)
- `example/ck_tile/50_sparse_attn/vsa_sparse_attention.cpp`  (+205/-0)
- `example/ck_tile/50_sparse_attn/jenga_sparse_attention.cpp`  (+199/-0)
- `example/ck_tile/50_sparse_attn/generate.py`  (+166/-0)
- `example/ck_tile/50_sparse_attn/CMakeLists.txt`  (+156/-0)
- `include/ck_tile/host/reference/reference_blocked_attention.hpp`  (+156/-0)
- `example/ck_tile/50_sparse_attn/codegen/cpp_symbol_map.py`  (+73/-0)

## Key added lines (kernel files)

**`example/ck_tile/50_sparse_attn/codegen/cpp_symbol_map.py`**
```
FWD_DTYPE_MAP = {
"fp16": "FmhaSparseFwdFp16",
"bf16": "FmhaSparseFwdBf16",
_MASK_SIMPLIFIED_MAP = {
```

**`example/ck_tile/50_sparse_attn/codegen/ops/fmha_fwd_jenga.py`**
```
import copy
from dataclasses import dataclass, field
import fnmatch
import itertools
```

**`example/ck_tile/50_sparse_attn/codegen/ops/fmha_fwd_vsa.py`**
```
import copy
from dataclasses import dataclass, field
import fnmatch
import itertools
```

**`example/ck_tile/50_sparse_attn/fmha_fwd_trek.hpp`**
```
namespace ck_tile {
inline bool is_load_tr_supported() { return is_gfx95_supported(); }
} // namespace ck_tile
struct FmhaSparseFwdFp16
```

**`example/ck_tile/50_sparse_attn/generate.py`**
```
import argparse
from enum import IntEnum
from pathlib import Path
import pkgutil
```
