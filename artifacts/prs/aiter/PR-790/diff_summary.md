# Diff summary

- **files changed:** 15
- **lines:** +730 / -146
- **kernel-ish files:** 12

## Files (by churn)

- `aiter/ops/triton/moe_op_mxfp4_silu_fused.py`  (+479/-0)
- `op_tests/triton_tests/test_moe_mx.py`  (+105/-48)
- `op_tests/op_benchmarks/triton/bench_moe_mx.py`  (+36/-50)
- `aiter/ops/triton/configs/moe/MI350X-MOE-MX_FP4.json`  (+35/-0)
- `op_tests/triton_tests/test_moe.py`  (+10/-24)
- `op_tests/op_benchmarks/triton/bench_moe.py`  (+22/-8)
- `aiter/ops/triton/utils/moe_common.py`  (+22/-0)
- `aiter/ops/triton/configs/moe/MI350X-MOE-FP8_W8A8.json`  (+5/-5)
- `aiter/ops/triton/moe_op_silu_fused.py`  (+4/-5)
- `aiter/ops/triton/utils/moe_config_utils.py`  (+7/-1)
- `aiter/ops/triton/moe_op.py`  (+1/-1)
- `op_tests/op_benchmarks/triton/bench_extend_attention.py`  (+1/-1)
- `op_tests/op_benchmarks/triton/bench_moe_align_block_size.py`  (+1/-1)
- `op_tests/op_benchmarks/triton/utils/benchmark_utils.py`  (+1/-1)
- `op_tests/op_benchmarks/triton/utils/model_configs.json`  (+1/-1)

## Key added lines (kernel files)

**`aiter/ops/triton/moe_op.py`**
```
output = torch.zeros(A.shape, device=A.device, dtype=B.dtype)
```

**`aiter/ops/triton/moe_op_mxfp4_silu_fused.py`**
```
import torch
import triton
import triton.language as tl
from typing import Any, Dict
```

**`aiter/ops/triton/moe_op_silu_fused.py`**
```
silu_acc = _silu_exp2(silu_acc)
silu_acc = _silu_exp2(silu_acc)
silu_acc = _silu_exp2(silu_acc)
output = torch.zeros(A.shape, device=A.device, dtype=B.dtype)
```

**`aiter/ops/triton/utils/moe_common.py`**
```
import torch
def torch_silu_and_mul_ref(input):
Performs the SiLU activation on the first half of the input tensor and
multiplies it element-wise with the second half.
```

**`aiter/ops/triton/utils/moe_config_utils.py`**
```
use_mxfp4: Optional[bool] = False,
elif use_mxfp4:
return "MX_FP4"
use_mxfp4: Optional[bool] = False,
```
