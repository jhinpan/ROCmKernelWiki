# Diff summary

- **files changed:** 10
- **lines:** +1583 / -2723
- **kernel-ish files:** 10

## Files (by churn)

- `aiter/ops/triton/_triton_kernels/flash_attn_triton_amd/utils.py`  (+155/-1403)
- `aiter/ops/triton/_triton_kernels/flash_attn_triton_amd/fwd_prefill.py`  (+283/-608)
- `aiter/ops/triton/_triton_kernels/flash_attn_triton_amd/common.py`  (+573/-0)
- `aiter/ops/triton/_triton_kernels/flash_attn_triton_amd/bwd.py`  (+204/-258)
- `aiter/ops/triton/_triton_kernels/flash_attn_triton_amd/fwd_decode.py`  (+193/-213)
- `aiter/ops/triton/_triton_kernels/flash_attn_triton_amd/interface_v3.py`  (+87/-183)
- `aiter/ops/triton/attention/mha_v3.py`  (+49/-29)
- `aiter/ops/triton/_triton_kernels/flash_attn_triton_amd/interface_v2.py`  (+29/-23)
- `setup.py`  (+9/-5)
- `aiter/ops/triton/_triton_kernels/attention/fav3_sage_attention.py`  (+1/-1)

## Key added lines (kernel files)

**`aiter/ops/triton/_triton_kernels/attention/fav3_sage_attention.py`**
```
from aiter.ops.triton._triton_kernels.flash_attn_triton_amd.common import (
```

**`aiter/ops/triton/_triton_kernels/flash_attn_triton_amd/bwd.py`**
```
import triton
import triton.language as tl
PREPROCESS_AUTOTUNE_KEYS = [
"max_seqlen_q",
```

**`aiter/ops/triton/_triton_kernels/flash_attn_triton_amd/common.py`**
```
Triton kernel helper functions shared across flash attention modules.
This module contains Triton JIT-compiled helper functions that are used within
the main attention kernels (fwd_prefill, fwd_decode, bwd). These are kept
separate from utils.py to allow stricter type checking on pure Python utilities.
```

**`aiter/ops/triton/_triton_kernels/flash_attn_triton_amd/fwd_decode.py`**
```
from .common import apply_rotary
get_shape_from_layout,
get_stride_from_layout,
FWD_DECODE_AUTOTUNE_KEYS = [
```

**`aiter/ops/triton/_triton_kernels/flash_attn_triton_amd/fwd_prefill.py`**
```
from .common import compute_alibi_block, compute_fp8_scaling_factors, apply_rotary
FWD_PREFILL_AUTOTUNE_KEYS = [
"IS_CAUSAL",
"dropout_p",
```
