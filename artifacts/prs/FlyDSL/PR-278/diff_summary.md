# Diff summary

- **files changed:** 35
- **lines:** +5359 / -6
- **kernel-ish files:** 31

## Files (by churn)

- `kernels/mxfp4_gemm_gfx1250.py`  (+989/-0)
- `kernels/mxfp8_gemm_gfx1250.py`  (+856/-0)
- `kernels/wmma_gemm_gfx1250.py`  (+677/-0)
- `python/flydsl/expr/rocdl/tdm_ops.py`  (+527/-0)
- `tests/kernels/test_mxfp4_gemm_gfx1250.py`  (+284/-0)
- `kernels/wmma_gemm_simple.py`  (+262/-0)
- `python/flydsl/expr/rocdl.py`  (+250/-0)
- `lib/Conversion/FlyToROCDL/FlyToROCDL.cpp`  (+214/-0)
- `tests/kernels/test_wmma_gemm_gfx1250.py`  (+191/-0)
- `python/flydsl/expr/rocdl/__init__.py`  (+186/-0)
- `lib/Dialect/FlyROCDL/GFX1250/MmaAtom.cpp`  (+159/-0)
- `tests/kernels/test_wmma_gemm_simple.py`  (+120/-0)
- `lib/CAPI/Dialect/FlyROCDL/FlyROCDLDialect.cpp`  (+115/-0)
- `python/flydsl/expr/gpu.py`  (+112/-1)
- `lib/Bindings/Python/FlyROCDLExtension.cpp`  (+76/-0)

## Key added lines (kernel files)

**`include/flydsl-c/FlyROCDLDialect.h`**
```
MLIR_CAPI_EXPORTED bool mlirTypeIsAFlyROCDLMmaAtomCDNA3_MFMAType(MlirType type);
MLIR_CAPI_EXPORTED MlirTypeID mlirFlyROCDLMmaAtomCDNA3_MFMATypeGetTypeID(void);
MLIR_CAPI_EXPORTED MlirType mlirFlyROCDLMmaAtomCDNA3_MFMATypeGet(int32_t m, int32_t n, int32_t k,
MlirType elemTyA, MlirType elemTyB,
```

**`kernels/mxfp4_gemm_gfx1250.py`**
```
"""MXFP4 GEMM kernel for gfx1250.
Uses V_WMMA_SCALE_F32_32X16X128_F4 with FP4 (E2M1) data and E8M0 block scales.
import flydsl.compiler as flyc
import flydsl.expr as fx
```

**`kernels/mxfp8_gemm_gfx1250.py`**
```
"""MXFP8 GEMM kernel for gfx1250. """
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
```

**`kernels/pipeline_utils.py`**
```
"""Shared pipeline utilities for gfx1250 GEMM kernels. """
def make_tail_plan(num_buffers, pre_loaded, extra):
"""Compute a compile-time tail execution plan for the N-stage pipeline.
Returns a list of (load_stage, compute_stage, outstanding) tuples, one per
```

**`kernels/wmma_gemm_gfx1250.py`**
```
"""TDM async copy WMMA GEMM kernel for gfx1250.
Supports double-buffer (2-stage) and triple-buffer (3-stage) pipelining
with TDM (Tensor Data Mover) hardware async copy for both A and B tiles.
import flydsl.compiler as flyc
```
