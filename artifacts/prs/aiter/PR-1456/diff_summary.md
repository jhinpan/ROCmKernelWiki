# Diff summary

- **files changed:** 7
- **lines:** +170 / -55
- **kernel-ish files:** 7

## Files (by churn)

- `aiter/ops/triton/gemm_a16w16_atomic.py`  (+51/-21)
- `aiter/ops/triton/gemm_afp4wfp4.py`  (+43/-11)
- `aiter/ops/triton/batched_gemm_a16wfp4.py`  (+29/-6)
- `aiter/ops/triton/gemm_a16wfp4.py`  (+27/-8)
- `aiter/ops/triton/gemm_afp4wfp4_pre_quant_atomic.py`  (+7/-6)
- `aiter/ops/triton/utils/common_utils.py`  (+9/-0)
- `aiter/ops/triton/batched_gemm_afp4wfp4_pre_quant.py`  (+4/-3)

## Key added lines (kernel files)

**`aiter/ops/triton/batched_gemm_a16wfp4.py`**
```
from aiter.ops.triton.utils.common_utils import deserialize_str
from aiter.jit.utils.torch_guard import torch_compile_guard
def batched_gemm_a16wfp4_fake_tensor(
x: torch.Tensor,
```

**`aiter/ops/triton/batched_gemm_afp4wfp4_pre_quant.py`**
```
from aiter.ops.triton.utils.common_utils import serialize_dict
config_hashable = serialize_dict(config) if config else None
x, w, w_scales, dtype, y, config_hashable, transpose_bm=False, prequant=True
```

**`aiter/ops/triton/gemm_a16w16_atomic.py`**
```
from aiter.ops.triton.utils.common_utils import serialize_dict, deserialize_str
from aiter.jit.utils.torch_guard import torch_compile_guard
def gemm_a16w16_atomic_fake_tensor(
x: torch.Tensor,
```

**`aiter/ops/triton/gemm_a16wfp4.py`**
```
from aiter.ops.triton.utils.common_utils import deserialize_str
from aiter.jit.utils.torch_guard import torch_compile_guard
def gemm_a16wfp4_fake_tensor(
x: torch.Tensor,
```

**`aiter/ops/triton/gemm_afp4wfp4.py`**
```
from aiter.ops.triton.utils.common_utils import serialize_dict, deserialize_str
from aiter.jit.utils.torch_guard import torch_compile_guard
def gemm_afp4wfp4_fake_tensor(
x: torch.Tensor,
```
