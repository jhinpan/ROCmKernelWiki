# Diff summary

- **files changed:** 8
- **lines:** +310 / -148
- **kernel-ish files:** 8

## Files (by churn)

- `tests/unit/test_expr_optional_rocdl.py`  (+140/-0)
- `python/flydsl/expr/rocdl/cluster.py`  (+117/-0)
- `python/flydsl/expr/gpu.py`  (+2/-113)
- `kernels/gemm_common_gfx1250.py`  (+15/-20)
- `python/flydsl/expr/__init__.py`  (+18/-3)
- `kernels/moe_gemm_2stage_mxscale_gfx1250.py`  (+10/-6)
- `kernels/gemm_fp8fp4_gfx1250.py`  (+4/-3)
- `kernels/wmma_gemm_gfx1250.py`  (+4/-3)

## Key added lines (kernel files)

**`kernels/gemm_common_gfx1250.py`**
```
"""Shared utilities for gfx1250 GEMM kernels (fp16 / mxfp4 / mxfp8)."""
from flydsl._mlir.dialects import llvm as llvm_dialect
from flydsl._mlir.dialects import scf
from flydsl.expr.rocdl import cluster
```

**`kernels/gemm_fp8fp4_gfx1250.py`**
```
from flydsl.expr.rocdl import cluster
local_x, local_y = cluster.compute_cluster_position()
a_mcast_mask, b_mcast_mask = cluster.compute_mcast_masks(local_x, local_y, cluster_m, cluster_n)
cluster.cluster_barrier()
```

**`kernels/moe_gemm_2stage_mxscale_gfx1250.py`**
```
from flydsl.expr.rocdl import cluster
_local_x, _local_y = cluster.compute_cluster_position()
_a_mcast_mask, b_mcast_mask = cluster.compute_mcast_masks(
cluster.cluster_barrier()
```

**`kernels/wmma_gemm_gfx1250.py`**
```
from flydsl.expr.rocdl import cluster
local_x, local_y = cluster.compute_cluster_position()
a_mcast_mask, b_mcast_mask = cluster.compute_mcast_masks(local_x, local_y, cluster_m, cluster_n)
cluster.cluster_barrier()
```

**`python/flydsl/expr/__init__.py`**
```
_LAZY_MODULES = {
"buffer_ops": ".buffer_ops",
"rocdl": ".rocdl",
"tdm_ops": ".rocdl.tdm_ops",
```
