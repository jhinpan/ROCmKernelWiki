# Diff summary

- **files changed:** 10
- **lines:** +1239 / -6
- **kernel-ish files:** 7

## Files (by churn)

- `aiter/ops/triton/_triton_kernels/fused_fp8_quant.py`  (+248/-0)
- `aiter/ops/triton/_triton_kernels/gemm_a16w8_blockscale.py`  (+237/-0)
- `aiter/ops/triton/fused_fp8_quant.py`  (+222/-0)
- `op_tests/triton_tests/test_gemm_a16w8_blockscale.py`  (+161/-0)
- `aiter/ops/triton/gemm_a16w8_blockscale.py`  (+142/-0)
- `op_tests/triton_tests/test_fused_fp8_quant.py`  (+104/-2)
- `aiter/ops/triton/configs/gemm/MI350X-GEMM-A16W8_BLOCKSCALE-N=7168-K=2048.json`  (+87/-0)
- `aiter/ops/triton/configs/gemm/MI300X-GEMM-A16W8_BLOCKSCALE.json`  (+14/-0)
- `aiter/ops/triton/configs/gemm/MI350X-GEMM-A16W8_BLOCKSCALE.json`  (+14/-0)
- `aiter/ops/triton/gemm_a8w8_blockscale.py`  (+10/-4)

## Key added lines (kernel files)

**`aiter/ops/triton/_triton_kernels/fused_fp8_quant.py`**
```
@triton.jit
def _fused_reduce_rms_fp8_group_quant_kernel(
inp1_ptr,
weight1_ptr,
```

**`aiter/ops/triton/_triton_kernels/gemm_a16w8_blockscale.py`**
```
import functools
import json
import os
import triton
```

**`aiter/ops/triton/fused_fp8_quant.py`**
```
_fused_reduce_rms_fp8_group_quant_kernel,
def fused_reduce_rms_fp8_group_quant(
inp1_weight,
inp1_epsilon,
```

**`aiter/ops/triton/gemm_a16w8_blockscale.py`**
```
from typing import Optional
import torch
import triton
from aiter.ops.triton._triton_kernels.gemm_a8w8_blockscale import (
```

**`aiter/ops/triton/gemm_a8w8_blockscale.py`**
```
skip_reduce: Optional[bool] = False,
if y is None and (config["NUM_KSPLIT"] == 1 or not skip_reduce):
y = torch.empty((M, N), dtype=dtype, device=x.device)
(config["NUM_KSPLIT"], M, N),
```
