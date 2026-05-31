# Diff summary

- **files changed:** 7 (diff was byte-capped; summary is partial)
- **lines:** +1659 / -1215
- **kernel-ish files:** 5

## Files (by churn)

- `aiter/ops/flydsl/kernels/mixed_moe_gemm_2stage.py`  (+1438/-973)
- `aiter/configs/model_configs/kimik2_fp4_tuned_fmoe.csv`  (+54/-165)
- `aiter/ops/flydsl/kernels/mfma_preshuffle_pipeline.py`  (+69/-25)
- `aiter/configs/model_configs/dsv3_fp4_tuned_fmoe.csv`  (+32/-47)
- `aiter/ops/flydsl/kernels/layout_utils.py`  (+39/-2)
- `aiter/fused_moe.py`  (+26/-2)
- `aiter/ops/flydsl/__init__.py`  (+1/-1)

## Key added lines (kernel files)

**`aiter/fused_moe.py`**
```
fuse_fp4_quant: bool = False
fuse_fp4_quant=False,
fuse_sort_scale=False,
_fq = fuse_fp4_quant or parsed.get("fuse_fp4_quant", False)
```

**`aiter/ops/flydsl/__init__.py`**
```
_REQUIRED_FLYDSL_VERSION = "0.1.1+20260401.5ac412e"
```

**`aiter/ops/flydsl/kernels/layout_utils.py`**
```
Optimisation: power-of-2 strides/shapes emit ``shrui`` / ``andi`` instead of
``divui`` / ``remui``, avoiding 10-15-cycle V_DIV sequences on CDNA GPUs.
import math as _math
def _is_pow2(n):
```

**`aiter/ops/flydsl/kernels/mfma_preshuffle_pipeline.py`**
```
Computes: col XOR ((row & (k_blocks16 - 1)) * 16)
k_blocks16 is always a power of 2 (tile_k_bytes / 16), so use
bitwise AND instead of remui to save ~10 VALU cycles on CDNA.
from flydsl.expr import arith as _swz_arith
```

**`aiter/ops/flydsl/kernels/mixed_moe_gemm_2stage.py`**
```
from .mfma_epilogues import c_shuffle_epilog
def _barrier(vmcnt=63, lgkmcnt=63):
"""Emit s_waitcnt + s_barrier via inline asm.
Bypasses LLVM SIInsertWaitcnts which would insert a conservative
```
