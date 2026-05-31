# Diff summary

- **files changed:** 7
- **lines:** +1809 / -3
- **kernel-ish files:** 7

## Files (by churn)

- `aiter/ops/triton/_triton_kernels/moe_op_gemm_a8w8.py`  (+515/-0)
- `aiter/ops/triton/moe_op_gemm_a8w8.py`  (+451/-0)
- `op_tests/triton_tests/test_moe_gemm_a8w8.py`  (+421/-0)
- `op_tests/op_benchmarks/triton/bench_moe_gemm_a8w8.py`  (+407/-0)
- `aiter/ops/triton/quant_moe.py`  (+11/-0)
- `op_tests/triton_tests/test_moe_routing.py`  (+3/-1)
- `aiter/ops/triton/moe_routing/topk.py`  (+1/-2)

## Key added lines (kernel files)

**`aiter/ops/triton/_triton_kernels/moe_op_gemm_a8w8.py`**
```
import torch
import triton
import triton.language as tl
from aiter.ops.triton.utils._triton.pid_preprocessing import pid_grid
```

**`aiter/ops/triton/moe_op_gemm_a8w8.py`**
```
from dataclasses import dataclass
import itertools
import sys
import torch
```

**`aiter/ops/triton/moe_routing/topk.py`**
```
(n_cols_pad, pids_x * TILE_SIZE), device=dev, dtype=torch.int32
```

**`aiter/ops/triton/quant_moe.py`**
```
def downcast_to_static_fp8_3d(x: torch.Tensor, scale: torch.Tensor):
assert x.ndim == 3
E, M, N = x.shape
x2d = x.reshape(E * M, N).contiguous()
```

**`op_tests/op_benchmarks/triton/bench_moe_gemm_a8w8.py`**
```
from itertools import chain
from pathlib import Path
from copy import deepcopy
import csv
```
