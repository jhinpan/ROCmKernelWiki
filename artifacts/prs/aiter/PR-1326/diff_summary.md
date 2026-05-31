# Diff summary

- **files changed:** 15
- **lines:** +3317 / -0
- **kernel-ish files:** 15

## Files (by churn)

- `aiter/ops/triton/_triton_kernels/moe_op_gemm_a8w4.py`  (+505/-0)
- `aiter/ops/triton/moe_op_gemm_a8w4.py`  (+438/-0)
- `aiter/ops/triton/_triton_kernels/quant_moe.py`  (+418/-0)
- `op_tests/triton_tests/test_moe_gemm_a8w4.py`  (+325/-0)
- `op_tests/op_benchmarks/triton/bench_moe_gemm_a8w4.py`  (+320/-0)
- `aiter/ops/triton/moe_routing/routing.py`  (+289/-0)
- `aiter/ops/triton/_triton_kernels/moe_routing/topk.py`  (+191/-0)
- `op_tests/triton_tests/test_moe_routing.py`  (+168/-0)
- `aiter/ops/triton/quant_moe.py`  (+159/-0)
- `aiter/ops/triton/_triton_kernels/moe_routing/routing.py`  (+150/-0)
- `aiter/ops/triton/_triton_kernels/moe_routing/bitmatrix.py`  (+104/-0)
- `aiter/ops/triton/moe_routing/topk.py`  (+84/-0)
- `aiter/ops/triton/_triton_kernels/moe_routing/expt_data.py`  (+83/-0)
- `aiter/ops/triton/moe_routing/bitmatrix.py`  (+82/-0)
- `aiter/ops/triton/utils/_triton/pid_preprocessing.py`  (+1/-0)

## Key added lines (kernel files)

**`aiter/ops/triton/_triton_kernels/moe_op_gemm_a8w4.py`**
```
import torch
import triton
import triton.language as tl
from aiter.ops.triton.utils._triton.pid_preprocessing import pid_grid
```

**`aiter/ops/triton/_triton_kernels/moe_routing/bitmatrix.py`**
```
import torch
import triton
import triton.language as tl
@triton.jit
```

**`aiter/ops/triton/_triton_kernels/moe_routing/expt_data.py`**
```
import triton
import triton.language as tl
@triton.jit
def _cdiv_pow2(n, log2_k):
```

**`aiter/ops/triton/_triton_kernels/moe_routing/routing.py`**
```
import triton
import triton.language as tl
from aiter.ops.triton._triton_kernels.moe_routing.expt_data import (
_expt_data_compute_stage1,
```

**`aiter/ops/triton/_triton_kernels/moe_routing/topk.py`**
```
import triton
import triton.language as tl
@triton.jit
def get_topmask_and_fullmask(x):
```
