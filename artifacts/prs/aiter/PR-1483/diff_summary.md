# Diff summary

- **files changed:** 9
- **lines:** +1888 / -8
- **kernel-ish files:** 9

## Files (by churn)

- `op_tests/op_benchmarks/triton/bench_moe_gemm_a8w8_blockscale.py`  (+477/-0)
- `aiter/ops/triton/_triton_kernels/moe/moe_op_gemm_a8w8_blockscale.py`  (+465/-0)
- `aiter/ops/triton/moe/moe_op_gemm_a8w8_blockscale.py`  (+441/-0)
- `op_tests/triton_tests/moe/test_moe_gemm_a8w8_blockscale.py`  (+343/-0)
- `aiter/ops/triton/moe/quant_moe.py`  (+58/-0)
- `op_tests/op_benchmarks/triton/bench_moe_gemm_a8w4.py`  (+42/-1)
- `op_tests/op_benchmarks/triton/bench_moe_gemm_a8w8.py`  (+42/-1)
- `aiter/ops/triton/_triton_kernels/moe/moe_op_gemm_a8w4.py`  (+10/-3)
- `aiter/ops/triton/_triton_kernels/moe/moe_op_gemm_a8w8.py`  (+10/-3)

## Key added lines (kernel files)

**`aiter/ops/triton/_triton_kernels/moe/moe_op_gemm_a8w4.py`**
```
def repr(s, x):
return f"{s}={x}" if x is not None else f"E_{len(hist)}({s})={n_rows}"
gindx = args.get("GatherIndx", None)
if gindx is not None:
```

**`aiter/ops/triton/_triton_kernels/moe/moe_op_gemm_a8w8.py`**
```
def repr(s, x):
return f"{s}={x}" if x is not None else f"E_{len(hist)}({s})={n_rows}"
gindx = args.get("GatherIndx", None)
if gindx is not None:
```

**`aiter/ops/triton/_triton_kernels/moe/moe_op_gemm_a8w8_blockscale.py`**
```
import torch
import triton
import triton.language as tl
from aiter.ops.triton.utils._triton.pid_preprocessing import pid_grid
```

**`aiter/ops/triton/moe/moe_op_gemm_a8w8_blockscale.py`**
```
import itertools
import torch
import triton
from aiter.ops.triton.moe.moe_routing.routing import RoutingData
```

**`aiter/ops/triton/moe/quant_moe.py`**
```
def dequant_x_blockscale(x, x_scales, per_row_x_scale, group_shape):
assert x_scales is not None
group_shape_m, _, group_shape_k = group_shape
M, K = x.shape
```
