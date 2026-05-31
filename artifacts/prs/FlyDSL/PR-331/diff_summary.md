# Diff summary

- **files changed:** 12
- **lines:** +1712 / -2425
- **kernel-ish files:** 12

## Files (by churn)

- `kernels/gemm_fp8fp4_gfx1250.py`  (+1112/-0)
- `kernels/mxfp4_gemm_gfx1250.py`  (+0/-989)
- `kernels/mxfp8_gemm_gfx1250.py`  (+0/-856)
- `tests/kernels/test_gemm_fp8fp4_gfx1250.py`  (+186/-94)
- `kernels/wmma_gemm_simple.py`  (+0/-262)
- `kernels/wmma_gemm_gfx1250.py`  (+120/-102)
- `kernels/gemm_common_gfx1250.py`  (+217/-0)
- `tests/kernels/test_wmma_gemm_simple.py`  (+0/-122)
- `python/flydsl/expr/rocdl/tdm_ops.py`  (+38/-0)
- `python/flydsl/expr/rocdl.py`  (+14/-0)
- `python/flydsl/expr/rocdl/__init__.py`  (+13/-0)
- `tests/kernels/utils/fp4_utils.py`  (+12/-0)

## Key added lines (kernel files)

**`kernels/gemm_common_gfx1250.py`**
```
"""Shared utilities for gfx1250 GEMM kernels (fp16 / mxfp4 / mxfp8). """
from flydsl._mlir import ir
from flydsl._mlir.dialects import llvm as llvm_dialect
from flydsl.expr import arith, buffer_ops, gpu, tdm_ops, vector
```

**`kernels/gemm_fp8fp4_gfx1250.py`**
```
"""Unified MXFP4/MXFP8/A8W4 GEMM kernel for gfx1250.
Supports FP4 (E2M1), FP8 (E4M3) and A8W4 (FP8 activation + FP4 weight)
data with E8M0 block scales via V_WMMA_SCALE instructions.
Select precision with ``data_format="fp4"|"fp8"|"a8w4"``.
```

**`kernels/wmma_gemm_gfx1250.py`**
```
from flydsl.utils.smem_allocator import SmemAllocator, SmemPtr
from kernels.gemm_common_gfx1250 import (
get_lds_memref, pipeline_fence,
store_acc_vec8_to_buffer, store_acc_vec8_to_lds,
```

**`python/flydsl/expr/rocdl.py`**
```
def disable_xdl_arb_stall():
"""Disable WMMA multicycle arbitration stall by setting SCHED_MODE bit 4."""
from .._mlir.dialects import llvm as _llvm
from . import arith as _arith
```

**`python/flydsl/expr/rocdl/__init__.py`**
```
def disable_xdl_arb_stall():
"""Disable WMMA multicycle arbitration stall by setting SCHED_MODE bit 4."""
from ..._mlir.dialects import llvm as _llvm
from .. import arith as _arith
```
