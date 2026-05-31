# Diff summary

- **files changed:** 102 (diff was byte-capped; summary is partial)
- **lines:** +5748 / -69
- **kernel-ish files:** 16

## Files (by churn)

- `aiter/ops/triton/_triton_kernels/unified_attention.py`  (+784/-0)
- `aiter/ops/triton/_triton_kernels/fused_kv_cache.py`  (+666/-13)
- `aiter/ops/triton/_triton_kernels/fused_gemm_a8w8_blockscale_a16w16.py`  (+452/-0)
- `aiter/ops/triton/_triton_kernels/fused_fp8_quant.py`  (+345/-0)
- `aiter/ops/triton/fused_fp8_quant.py`  (+336/-0)
- `aiter/ops/triton/_triton_kernels/gemm_afp4wfp4.py`  (+262/-22)
- `aiter/ops/triton/_triton_kernels/fused_qkv_split_qk_rope.py`  (+187/-0)
- `aiter/ops/triton/configs/gemm/MI350X-GEMM-A16W16-N=128-K=2880.json`  (+145/-0)
- `aiter/ops/triton/configs/gemm/MI350X-GEMM-A16W16-N=2280-K=512.json`  (+145/-0)
- `aiter/ops/triton/configs/gemm/MI350X-GEMM-A16W16-N=2880-K=4096.json`  (+145/-0)
- `aiter/ops/triton/configs/gemm/MI350X-GEMM-A16W16-N=5120-K=2880.json`  (+145/-0)
- `aiter/ops/triton/configs/gemm/MI350X-GEMM-A16W16-N=640-K=2880.json`  (+145/-0)
- `aiter/ops/triton/configs/gemm/MI350X-FUSED-GEMM-A8W8_BLOCKSCALE-A16W16-N8=512-N16=256-K=7168.json`  (+110/-0)
- `aiter/ops/triton/configs/gemm/MI350X-GEMM-AFP4WFP4_PRESHUFFLED.json`  (+87/-0)
- `aiter/ops/triton/configs/gemm/MI350X-GEMM-AFP4WFP4_PRESHUFFLED-N=10240-K=8192.json`  (+86/-0)

## Key added lines (kernel files)

**`aiter/ops/triton/_triton_kernels/activation.py`**
```
from .fused_fp8_quant import _fp8_quant_op
@triton.heuristics(
"EVEN_N": lambda args: args["N"] % args["BLOCK_SIZE_N"] == 0,
@triton.jit
```

**`aiter/ops/triton/_triton_kernels/fused_add_rmsnorm_pad.py`**
```
import triton
import triton.language as tl
@triton.jit
def _rmsmorm_op(row, weight, n_cols, epsilon):
```

**`aiter/ops/triton/_triton_kernels/fused_fp8_quant.py`**
```
import triton
import triton.language as tl
@triton.jit
def _rmsmorm_op(row, weight, n_cols, epsilon):
```

**`aiter/ops/triton/_triton_kernels/fused_gemm_a8w8_blockscale_a16w16.py`**
```
from typing import Optional
import functools
import json
import os
```

**`aiter/ops/triton/_triton_kernels/fused_kv_cache.py`**
```
from aiter.ops.triton.rope import _get_gptj_rotated_x_1D, _get_neox_rotated_x_1D
@triton.jit
def _unit_cat(
x_out_ptr,
```
