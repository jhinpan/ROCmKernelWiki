# Diff summary

- **files changed:** 15
- **lines:** +208 / -96
- **kernel-ish files:** 15

## Files (by churn)

- `aiter/jit/core.py`  (+46/-23)
- `aiter/jit/utils/torch_guard.py`  (+57/-0)
- `aiter/ops/gemm_op_a8w8.py`  (+18/-37)
- `aiter/ops/gemm_op_a4w4.py`  (+40/-9)
- `aiter/ops/batched_gemm_op_bf16.py`  (+19/-6)
- `aiter/jit/utils/chip_info.py`  (+8/-2)
- `csrc/ck_gemm_a4w4_blockscale/gemm_a4w4_blockscale_tune.py`  (+6/-4)
- `csrc/ck_batched_gemm_bf16/batched_gemm_bf16.cu`  (+2/-2)
- `csrc/ck_batched_gemm_bf16/batched_gemm_bf16_tune.cu`  (+2/-2)
- `csrc/ck_batched_gemm_bf16/include/batched_gemm_bf16.h`  (+2/-2)
- `csrc/ck_gemm_a4w4_blockscale/gemm_a4w4_blockscale.cu`  (+2/-2)
- `csrc/ck_gemm_a4w4_blockscale/include/gemm_a4w4_blockscale.h`  (+2/-2)
- `csrc/py_itfs_cu/asm_gemm_a4w4.cu`  (+2/-2)
- `op_tests/test_gemm_a4w4.py`  (+1/-2)
- `csrc/include/asm_gemm_a4w4.h`  (+1/-1)

## Key added lines (kernel files)

**`aiter/jit/core.py`**
```
SPECIAL_OPS_MUTATES_ARGS = {
"topk_softmax": [
],  # "topk_weights", "topk_indices", "token_expert_indices"
"biased_grouped_topk_hip": ["topk_weights", "topk_ids"],
```

**`aiter/jit/utils/chip_info.py`**
```
from torch_guard import torch_compile_guard
@torch_compile_guard()
def get_cu_num_custom_op() -> int:
@functools.lru_cache(maxsize=1)
```

**`aiter/jit/utils/torch_guard.py`**
```
aiter_lib = None
def torch_compile_guard(mutates_args: list[str] = [], device: str = "cpu"):
def decorator(func):
import torch
```

**`aiter/ops/batched_gemm_op_bf16.py`**
```
def gen_batched_gemm_bf16_tune_fake_tensor(
XQ: Tensor, WQ: Tensor, out: Tensor, kernelId: int, splitK: int = 0
) -> Tensor:
return out
```

**`aiter/ops/gemm_op_a4w4.py`**
```
return gemm_a4w4_blockscale(A, B, A_scale, B_scale, out, splitK=splitK)
return gemm_a4w4_asm(
def gen_gemm_a4w4_asm_fake_tensors(
A: Tensor,  # A:[M, K/2] f4x2
```
