# Diff summary

- **files changed:** 5 (diff was byte-capped; summary is partial)
- **lines:** +3284 / -1357
- **kernel-ish files:** 5

## Files (by churn)

- `kernels/mixed_moe_gemm_2stage.py`  (+2738/-1256)
- `kernels/mfma_preshuffle_pipeline.py`  (+142/-85)
- `kernels/mfma_epilogues.py`  (+187/-15)
- `kernels/layout_utils.py`  (+181/-0)
- `kernels/kernels_common.py`  (+36/-1)

## Key added lines (kernel files)

**`kernels/kernels_common.py`**
```
from contextlib import contextmanager
from flydsl._mlir.dialects import arith as _std_arith, builtin, gpu as _gpu, llvm as _llvm, scf as _scf
@contextmanager
def _if_then(if_op, scf=None):
```

**`kernels/layout_utils.py`**
```
"""Layout helpers for GEMM kernels.
Parses fly layout type strings (e.g. '(4,64):(64,1)') and computes
idx2crd / crd2idx with plain arith ops for static layouts.
Falls back to fly dialect ops for dynamic layouts.
```

**`kernels/mfma_epilogues.py`**
```
When ``lds_out_split`` is provided, the epilogue runs in split-LDS mode:
waves are partitioned into two groups (group A uses ``lds_out``, group B
uses ``lds_out_split``), each handling half of the N dimension.
from flydsl._mlir.dialects.arith import CmpIPredicate
```

**`kernels/mfma_preshuffle_pipeline.py`**
```
Computes: col XOR ((row & (k_blocks16 - 1)) * 16)
k_blocks16 is always a power of 2 (tile_k_bytes / 16), so use
bitwise AND instead of remui to save ~10 VALU cycles on CDNA.
from flydsl.expr import arith as _swz_arith
```

**`kernels/mixed_moe_gemm_2stage.py`**
```
This module contains the **kernel builder code** for:
- `moe_gemm1` (stage1, with silu/swiglu activation)
- `moe_gemm2` (stage2)
It is extracted from `tests/kernels/test_moe_gemm.py` so that:
```
