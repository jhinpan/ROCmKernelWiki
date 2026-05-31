# Diff summary

- **files changed:** 8 (diff was byte-capped; summary is partial)
- **lines:** +3461 / -620
- **kernel-ish files:** 8

## Files (by churn)

- `kernels/small_m_hgemm.py`  (+1249/-0)
- `kernels/qk_norm_rope_quant.py`  (+955/-0)
- `kernels/silu_and_mul_fq.py`  (+362/-242)
- `kernels/moe_gemm_2stage.py`  (+350/-210)
- `kernels/mixed_moe_gemm_2stage.py`  (+137/-124)
- `kernels/splitk_hgemm.py`  (+256/-0)
- `kernels/preshuffle_gemm.py`  (+128/-44)
- `kernels/moe_common.py`  (+24/-0)

## Key added lines (kernel files)

**`kernels/mixed_moe_gemm_2stage.py`**
```
"""MoE GEMM stage1/stage2 kernel implementations (FLIR MFMA FP8/FP16).
This module intentionally contains the **kernel builder code** for:
- `moe_gemm1` (stage1)
from contextlib import contextmanager
```

**`kernels/moe_common.py`**
```
"""Common types shared across MoE FlyDSL kernel modules."""
from enum import Enum
class GateMode(str, Enum):
"""Gate/Up computation strategy for stage1 GEMM.
```

**`kernels/moe_gemm_2stage.py`**
```
from .mfma_epilogues import c_shuffle_epilog, default_epilog, mfma_epilog
from .mfma_preshuffle_pipeline import (
def out_mlir():
return (lambda ty: ty() if callable(ty) else ty)(T.f16 if out_dtype == "f16" else T.bf16)
```

**`kernels/preshuffle_gemm.py`**
```
import functools
from .mfma_epilogues import mfma_epilog
from .mfma_preshuffle_pipeline import (
@functools.lru_cache(maxsize=1024)
```

**`kernels/qk_norm_rope_quant.py`**
```
"""Fused per-token RMSNorm + GPT-J RoPE + optional FP8 quant (FlyDSL).
Q + KV combined into a single kernel launch (grid Y = num_tokens, grid X =
num_q_heads + 1: bid_x ∈ [0, H) handle Q heads, bid_x == H handles KV).
Hard-coded MVP shape: D=512, RD=64, BLOCK_THREADS=64. Each block uses one
```
