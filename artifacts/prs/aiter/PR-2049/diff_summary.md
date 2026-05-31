# Diff summary

- **files changed:** 12
- **lines:** +1951 / -92
- **kernel-ish files:** 12

## Files (by churn)

- `aiter/ops/triton/moe/moe_op_gemm_int8_smoothquant.py`  (+467/-0)
- `aiter/ops/triton/_triton_kernels/moe/moe_op_gemm_int8_smoothquant.py`  (+416/-0)
- `op_tests/op_benchmarks/triton/bench_moe_gemm_int8_smoothquant.py`  (+325/-0)
- `op_tests/triton_tests/moe/test_moe_gemm_int8_smoothquant.py`  (+286/-0)
- `aiter/ops/triton/_triton_kernels/moe/quant_moe.py`  (+247/-0)
- `aiter/ops/triton/moe/quant_moe.py`  (+133/-2)
- `aiter/ops/triton/_triton_kernels/moe/moe_routing/topk.py`  (+33/-40)
- `aiter/ops/triton/moe/moe_routing/routing.py`  (+11/-13)
- `aiter/ops/triton/_triton_kernels/moe/moe_routing/routing.py`  (+14/-9)
- `aiter/ops/triton/moe/moe_routing/topk.py`  (+11/-12)
- `op_tests/triton_tests/moe/test_moe_routing.py`  (+7/-15)
- `op_tests/op_benchmarks/triton/bench_moe_gemm_a4w4.py`  (+1/-1)

## Key added lines (kernel files)

**`aiter/ops/triton/_triton_kernels/moe/moe_op_gemm_int8_smoothquant.py`**
```
import torch
import triton
import triton.language as tl
from aiter.ops.triton.utils._triton.pid_preprocessing import pid_grid
```

**`aiter/ops/triton/_triton_kernels/moe/moe_routing/routing.py`**
```
N_EXPTS_ACT_PAD: tl.constexpr,
tl.static_assert(N_EXPTS_ACT_PAD * BLOCK_M <= 32768)
local_offs = tl.arange(0, N_EXPTS_ACT_PAD * BLOCK_M)
if EVEN_M and N_EXPTS_ACT == N_EXPTS_ACT_PAD:
```

**`aiter/ops/triton/_triton_kernels/moe/moe_routing/topk.py`**
```
N_EXPTS_ACT_PAD: tl.constexpr,
x = (x.to(x_ultype) << 16) | offs_x_n[None, :]
acc = tl.topk(x, N_EXPTS_ACT_PAD, dim=1)
x = (x.to(x_ultype) << 16) | offs_x_n[None, :]
```

**`aiter/ops/triton/_triton_kernels/moe/quant_moe.py`**
```
@triton.jit
def _smoothquant_fuse_quant_kernel(
X_ptr,  # bf16 input [M, K]
stride_x_m,
```

**`aiter/ops/triton/moe/moe_op_gemm_int8_smoothquant.py`**
```
import itertools
import torch
import triton
from aiter.ops.triton.moe.moe_routing.routing import RoutingData
```
