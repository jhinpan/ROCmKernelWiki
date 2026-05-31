# Diff summary

- **files changed:** 8
- **lines:** +1130 / -3
- **kernel-ish files:** 8

## Files (by churn)

- `aiter/ops/triton/_triton_kernels/fusions/fused_reduce_qk_norm_rope_swa_write.py`  (+289/-0)
- `op_tests/triton_tests/fusions/test_fused_reduce_qk_norm_rope_swa_write.py`  (+249/-0)
- `aiter/ops/triton/fusions/fused_reduce_qk_norm_rope_swa_write.py`  (+191/-0)
- `aiter/ops/triton/fusions/fused_clamp_act_mul.py`  (+171/-0)
- `aiter/ops/triton/_triton_kernels/fusions/fused_clamp_act_mul.py`  (+123/-0)
- `op_tests/triton_tests/fusions/test_fused_clamp_act_mul.py`  (+103/-0)
- `aiter/ops/triton/_triton_kernels/quant/fused_mxfp4_quant.py`  (+4/-2)
- `op_tests/triton_tests/gemm/basic/test_gemm_a8w8_blockscale.py`  (+0/-1)

## Key added lines (kernel files)

**`aiter/ops/triton/_triton_kernels/fusions/fused_clamp_act_mul.py`**
```
"""Fused SwiGLU clamp + SiLU * up + optional token weights + optional per-row FP8 group quant (128).
Each program handles one row (token). ``inp`` is ``[M, 2 * N]`` with gate in the first
``N`` columns and up in the second ``N`` (same layout as ``torch.chunk(2, dim=-1)``).
Gate clamp matches DeepSeek-V4 reference: ``clamp(gate, max=limit)`` only; up uses
```

**`aiter/ops/triton/_triton_kernels/fusions/fused_reduce_qk_norm_rope_swa_write.py`**
```
"""Fused split-K reduce + per-head weighted RMSNorm + RoPE (tail) on Q,
per-row weighted RMSNorm + RoPE (tail) on KV (+ optional SWA KV write).
Grid: ``(cdiv(M, BLOCK_SIZE_M), num_local_heads + 1)``. Each program tile
handles ``BLOCK_SIZE_M`` tokens. Programs with ``pid_h < num_local_heads``
```

**`aiter/ops/triton/_triton_kernels/quant/fused_mxfp4_quant.py`**
```
if weight is not None:
rms_norm = row * norm_factor[:, None] * weight
rms_norm = row * norm_factor[:, None]
```

**`aiter/ops/triton/fusions/fused_clamp_act_mul.py`**
```
from __future__ import annotations
from typing import Literal, Optional
import torch
import triton
```

**`aiter/ops/triton/fusions/fused_reduce_qk_norm_rope_swa_write.py`**
```
from typing import Optional
import torch
import triton
from aiter.ops.triton._triton_kernels.fusions.fused_reduce_qk_norm_rope_swa_write import (
```
