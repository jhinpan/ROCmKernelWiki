# Diff summary

- **files changed:** 6
- **lines:** +415 / -59
- **kernel-ish files:** 6

## Files (by churn)

- `aiter/ops/triton/_gluon_kernels/gfx942/moe/moe_op_gemm_int8_smoothquant.py`  (+273/-0)
- `aiter/ops/triton/moe/moe_op_gemm_int8_smoothquant.py`  (+137/-54)
- `aiter/ops/triton/_triton_kernels/moe/moe_op_gemm_int8_smoothquant.py`  (+5/-5)
- `aiter/ops/triton/_gluon_kernels/__init__.py`  (+0/-0)
- `aiter/ops/triton/_gluon_kernels/gfx942/__init__.py`  (+0/-0)
- `aiter/ops/triton/_gluon_kernels/gfx942/moe/__init__.py`  (+0/-0)

## Key added lines (kernel files)

**`aiter/ops/triton/_gluon_kernels/gfx942/moe/moe_op_gemm_int8_smoothquant.py`**
```
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
@triton.heuristics(
```

**`aiter/ops/triton/_triton_kernels/moe/moe_op_gemm_int8_smoothquant.py`**
```
Y = (X * diag(s)^-1) @ (diag(s) * W)
- X is int8 activations [M, K] (quantized X * diag(s)^-1)
- W is int8 weights [E, K, N] (quantized diag(s) * W)
- x_scale is fp32 per-token scale [M] (dequant scale for X)
```

**`aiter/ops/triton/moe/moe_op_gemm_int8_smoothquant.py`**
```
from aiter.ops.triton._gluon_kernels.gfx942.moe.moe_op_gemm_int8_smoothquant import (
_gluon_moe_gemm_int8_smoothquant,
from aiter.ops.triton.utils._triton import arch_info
kpack = 2 if arch_info.get_arch() == "gfx942" else 1
```
