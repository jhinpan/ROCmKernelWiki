# Diff summary

- **files changed:** 7 (diff was byte-capped; summary is partial)
- **lines:** +5817 / -8
- **kernel-ish files:** 7

## Files (by churn)

- `kernels/moe_gemm_2stage_mxscale_gfx1250.py`  (+2889/-0)
- `kernels/moe_gemm_2stage_common_gfx1250.py`  (+1089/-0)
- `kernels/moe_gemm_2stage_wmma_gfx1250.py`  (+912/-0)
- `tests/kernels/benchmark_common.py`  (+443/-0)
- `python/flydsl/expr/rocdl/tdm_ops.py`  (+324/-0)
- `tests/kernels/test_moe_gemm_mxscale_gfx1250.py`  (+136/-0)
- `kernels/gemm_common_gfx1250.py`  (+24/-8)

## Key added lines (kernel files)

**`kernels/gemm_common_gfx1250.py`**
```
from flydsl._mlir.dialects import llvm as llvm_dialect, scf
def workgroup_barrier(use_cluster=False):
"""Issue the appropriate barrier for LDS visibility.
Cluster mode layers an inter-workgroup barrier on top of the regular
```

**`kernels/moe_gemm_2stage_common_gfx1250.py`**
```
"""Shared utilities for gfx1250 MoE 2-stage kernels.
Common helpers used by both the fp16 WMMA kernels and the mxscale
(fp4/fp8/a8w4) kernels.
from __future__ import annotations
```

**`kernels/moe_gemm_2stage_mxscale_gfx1250.py`**
```
"""gfx1250 MoE 2-stage mxscale kernels (fp4/fp8/a8w4).
Implements stage1/stage2 single-kernel inline paths using the
``wmma_scale_f32_16x16x128_f8f6f4`` and ``wmma_scale_f32_32x16x128_f4``
instructions for microscaling block formats with E8M0 scales.
```

**`kernels/moe_gemm_2stage_wmma_gfx1250.py`**
```
"""gfx1250 MoE 2-stage fp16 WMMA kernels.
Implements stage1/stage2 single-kernel inline paths using the
``wmma_f32_16x16x32_f16`` instruction for fp16 (and bf16 via host
conversion) inputs.
```

**`python/flydsl/expr/rocdl/tdm_ops.py`**
```
"TDMGatherDescriptor",
"make_tensor_gather_dgroup0",
"make_tensor_gather_descriptor",
"tensor_load_gather",
```
