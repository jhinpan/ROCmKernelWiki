# Diff summary

- **files changed:** 6
- **lines:** +2504 / -872
- **kernel-ish files:** 6

## Files (by churn)

- `kernels/moe_gemm_2stage_mxscale_gfx1250.py`  (+1509/-345)
- `tests/kernels/test_moe_gemm_mxscale_gfx1250.py`  (+536/-351)
- `kernels/moe_gemm_2stage_common_gfx1250.py`  (+347/-16)
- `tests/kernels/test_moe_gemm_wmma_gfx1250.py`  (+90/-142)
- `kernels/moe_gemm_2stage_wmma_gfx1250.py`  (+10/-14)
- `tests/kernels/benchmark_common.py`  (+12/-4)

## Key added lines (kernel files)

**`kernels/moe_gemm_2stage_common_gfx1250.py`**
```
def _finalize_alloc_and_launch_2d(*, ctx, alloc, launcher, gx, gy, block_threads: int, stream, waves_per_eu, ir,
cluster=None, gz=1):
for op in ctx.gpu_module_body.operations:
if hasattr(op, "attributes") and op.OPERATION_NAME == "gpu.func":
```

**`kernels/moe_gemm_2stage_mxscale_gfx1250.py`**
```
_emit_stage1_gate_up_splitk_epilogue,
_emit_swiglu,
use_tdm_gather_as: bool = True,
k_batch: int = 1,
```

**`kernels/moe_gemm_2stage_wmma_gfx1250.py`**
```
waves_per_eu=waves_per_eu,
_keep_const_expr_ref = const_expr  # noqa: F841
waves_per_eu=waves_per_eu,
```

**`tests/kernels/benchmark_common.py`**
```
def speedup_aiter_vs_flydsl(self) -> Optional[float]:
return self.flydsl_gpu_us / self.aiter_gpu_us
sp = r.speedup_aiter_vs_flydsl
if tile_k1 is None:
```

**`tests/kernels/test_moe_gemm_mxscale_gfx1250.py`**
```
import argparse
from typing import Optional, Tuple
from tests.test_common import verify_output
bench_bytes_moved_stage1 as _bench_bytes_moved_stage1,
```
