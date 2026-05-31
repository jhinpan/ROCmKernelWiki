# Diff summary

- **files changed:** 10
- **lines:** +49 / -18
- **kernel-ish files:** 10

## Files (by churn)

- `aiter/ops/triton/moe_op_silu_fused.py`  (+10/-4)
- `aiter/ops/triton/moe_op.py`  (+9/-4)
- `aiter/ops/triton/moe_op_e2e.py`  (+5/-2)
- `aiter/ops/triton/moe_op_gelu.py`  (+5/-2)
- `aiter/ops/triton/gemm_a8w8.py`  (+3/-2)
- `aiter/ops/triton/mha_fused_bwd.py`  (+4/-1)
- `aiter/ops/triton/utils/arch_info.py`  (+5/-0)
- `aiter/ops/triton/extend_attention.py`  (+3/-1)
- `aiter/ops/triton/moe_op_mxfp4.py`  (+3/-1)
- `aiter/ops/triton/mha.py`  (+2/-1)

## Key added lines (kernel files)

**`aiter/ops/triton/extend_attention.py`**
```
from aiter.ops.triton.utils.arch_info import get_num_xcds
NUM_XCDS: tl.constexpr,
NUM_XCDS=get_num_xcds(),
```

**`aiter/ops/triton/gemm_a8w8.py`**
```
from aiter.ops.triton.utils.arch_info import get_num_xcds
NUM_XCDS: tl.constexpr,
NUM_XCDS=get_num_xcds(),
```

**`aiter/ops/triton/mha.py`**
```
from aiter.ops.triton.utils.arch_info import get_num_xcds
NUM_XCD=get_num_xcds(),
```

**`aiter/ops/triton/mha_fused_bwd.py`**
```
from aiter.ops.triton.utils.arch_info import get_num_xcds
NUM_XCD: tl.constexpr,
NUM_XCD=get_num_xcds(),
NUM_XCD=get_num_xcds(),
```

**`aiter/ops/triton/moe_op.py`**
```
from aiter.ops.triton.utils.arch_info import get_num_xcds
NUM_XCDS: tl.constexpr,
NUM_XCDS: tl.constexpr,
NUM_XCDS: tl.constexpr,
```
