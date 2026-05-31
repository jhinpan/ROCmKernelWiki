# Diff summary

- **files changed:** 8
- **lines:** +1433 / -0
- **kernel-ish files:** 6

## Files (by churn)

- `aiter/ops/triton/_triton_kernels/fused_gemm_a8w8_blockscale_a16w16.py`  (+452/-0)
- `op_tests/triton_tests/test_fused_fp8_quant.py`  (+219/-0)
- `aiter/ops/triton/fused_gemm_a8w8_blockscale_a16w16.py`  (+210/-0)
- `aiter/ops/triton/_triton_kernels/fused_fp8_quant.py`  (+149/-0)
- `aiter/ops/triton/fused_fp8_quant.py`  (+143/-0)
- `op_tests/triton_tests/test_fused_gemm_a8w8_blockscale_a16w16.py`  (+136/-0)
- `aiter/ops/triton/configs/gemm/MI350X-FUSED-GEMM-A8W8_BLOCKSCALE-A16W16-N8=512-N16=256-K=7168.json`  (+110/-0)
- `aiter/ops/triton/configs/gemm/MI350X-FUSED-GEMM-A8W8_BLOCKSCALE-A16W16.json`  (+14/-0)

## Key added lines (kernel files)

**`aiter/ops/triton/_triton_kernels/fused_fp8_quant.py`**
```
@triton.jit
def _fused_reduce_act_mul_fp8_group_quant(
y_scale_ptr,
stride_x_spk,
```

**`aiter/ops/triton/_triton_kernels/fused_gemm_a8w8_blockscale_a16w16.py`**
```
from typing import Optional
import functools
import json
import os
```

**`aiter/ops/triton/fused_fp8_quant.py`**
```
from typing import Optional
_fused_reduce_act_mul_fp8_group_quant,
from aiter.ops.triton._triton_kernels.activation import (
_get_activation_from_str,
```

**`aiter/ops/triton/fused_gemm_a8w8_blockscale_a16w16.py`**
```
from typing import Optional
import functools
import json
import os
```

**`op_tests/triton_tests/test_fused_fp8_quant.py`**
```
import torch
import pytest
from aiter.ops.triton.fused_fp8_quant import (
fused_rms_fp8_group_quant,
```
