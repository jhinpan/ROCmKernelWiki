# Diff summary

- **files changed:** 11
- **lines:** +45 / -27
- **kernel-ish files:** 11

## Files (by churn)

- `aiter/ops/gemm_op_a8w8.py`  (+19/-8)
- `aiter/jit/core.py`  (+7/-1)
- `aiter/ops/moe_op.py`  (+4/-4)
- `csrc/include/rocm_ops.hpp`  (+4/-4)
- `csrc/kernels/topk_softmax_kernels_group.cu`  (+2/-3)
- `aiter/ops/attention.py`  (+2/-2)
- `csrc/py_itfs_cu/asm_pa.cu`  (+2/-1)
- `op_tests/test_moeTopkSoftmax.py`  (+2/-1)
- `aiter/ops/topk.py`  (+1/-1)
- `csrc/include/attention_asm.h`  (+1/-1)
- `csrc/include/moe_op.h`  (+1/-1)

## Key added lines (kernel files)

**`aiter/jit/core.py`**
```
if hasattr(torch.library, "infer_schema"):
sig = torch.library.infer_schema(func, mutates_args="unknown")
import torch._custom_op.impl
sig = torch._custom_op.impl.infer_schema(func, mutates_args)
```

**`aiter/ops/attention.py`**
```
kernelName: Optional[str] = None,
kernelName: Optional[str] = None,
```

**`aiter/ops/gemm_op_a8w8.py`**
```
@functools.lru_cache(maxsize=1024)
def get_CKGEMM_config(M: int, N: int, K: int, tuned_file="a8w8_tuned_gemm.csv"):
import torch
op_name = "aiter::get_CKGEMM_config_"
```

**`aiter/ops/moe_op.py`**
```
kernelName: Optional[str] = None,
kernelName: Optional[str] = None,
kernelName: Optional[str] = None,
kernelName: Optional[str] = None,
```

**`aiter/ops/topk.py`**
```
scoring_func: bool = True,
```
