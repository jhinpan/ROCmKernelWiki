# Diff summary

- **files changed:** 5 (diff was byte-capped; summary is partial)
- **lines:** +7526 / -0
- **kernel-ish files:** 4

## Files (by churn)

- `aiter/ops/triton/_triton_kernels/flash_attn_triton_amd/bwd.py`  (+4941/-0)
- `aiter/ops/triton/_triton_kernels/flash_attn_triton_amd/fwd_decode.py`  (+1404/-0)
- `aiter/ops/triton/_triton_kernels/flash_attn_triton_amd/fwd_prefill.py`  (+1173/-0)
- `.gitignore`  (+4/-0)
- `aiter/ops/triton/_triton_kernels/flash_attn_triton_amd/__init__.py`  (+4/-0)

## Key added lines (kernel files)

**`aiter/ops/triton/_triton_kernels/flash_attn_triton_amd/__init__.py`**
```
from . import interface_v2 as flash_attn_2
from . import interface_v3 as flash_attn_3
__all__ = ["flash_attn_2", "flash_attn_3"]
```

**`aiter/ops/triton/_triton_kernels/flash_attn_triton_amd/bwd.py`**
```
import os
import torch
import triton  # type: ignore
import triton.language as tl  # type: ignore
```

**`aiter/ops/triton/_triton_kernels/flash_attn_triton_amd/fwd_decode.py`**
```
import os
import warnings
import torch
import triton
```

**`aiter/ops/triton/_triton_kernels/flash_attn_triton_amd/fwd_prefill.py`**
```
import os
import warnings
import torch
import triton
```
