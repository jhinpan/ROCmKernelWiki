# Diff summary

- **files changed:** 10 (diff was byte-capped; summary is partial)
- **lines:** +5683 / -31
- **kernel-ish files:** 8

## Files (by churn)

- `aiter/ops/flydsl/kernels/mixed_moe_gemm_2stage.py`  (+2825/-0)
- `aiter/ops/flydsl/kernels/moe_gemm_2stage.py`  (+1835/-0)
- `aiter/ops/flydsl/kernels/mfma_preshuffle_pipeline.py`  (+543/-0)
- `aiter/ops/flydsl/kernels/mfma_epilogues.py`  (+293/-0)
- `aiter/fused_moe.py`  (+119/-26)
- `aiter/ops/flydsl/__init__.py`  (+25/-0)
- `aiter/configs/model_configs/dsv3_fp4_tuned_fmoe.csv`  (+15/-0)
- `aiter/jit/core.py`  (+10/-5)
- `README.md`  (+14/-0)
- `aiter/ops/flydsl/kernels/__init__.py`  (+4/-0)

## Key added lines (kernel files)

**`aiter/fused_moe.py`**
```
from aiter.ops.flydsl.utils import is_flydsl_available
def _flydsl_stage2_wrapper(
inter_states,
sorted_token_ids,
```

**`aiter/jit/core.py`**
```
base_cols = [c for c in df_list[0].columns if c != "_tag"]
new_cols = [c for c in df.columns if c != "_tag"]
base_cols == new_cols
), f"Column mismatch between {path_list[0]} and {path}, {base_cols}, {new_cols}"
```

**`aiter/ops/flydsl/__init__.py`**
```
"""FlyDSL -- high-performance GPU kernels implemented using FlyDSL.
Kernel compilation and public APIs are only available when the ``flydsl``
package is installed.  Use ``is_flydsl_available()`` to check at runtime.
from .utils import is_flydsl_available
```

**`aiter/ops/flydsl/kernels/__init__.py`**
```
"""FlyDSL MOE kernel builders (stage1, stage2, reduction).
Internal sub-package -- use ``aiter.ops.flydsl`` as the public entry point.
```

**`aiter/ops/flydsl/kernels/mfma_epilogues.py`**
```
"""Reusable epilogue helpers for MFMA 16x16-based kernels.
This module provides:
- `mfma_epilog(...)`
A single entrypoint that dispatches to either the default row-epilogue or the
```
