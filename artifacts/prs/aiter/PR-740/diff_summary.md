# Diff summary

- **files changed:** 41
- **lines:** +1794 / -582
- **kernel-ish files:** 41

## Files (by churn)

- `aiter/ops/mha.py`  (+855/-359)
- `aiter/jit/core.py`  (+186/-2)
- `aiter/ops/gemm_op_a8w8.py`  (+154/-18)
- `aiter/ops/moe_op.py`  (+99/-42)
- `aiter/ops/custom_all_reduce.py`  (+97/-11)
- `aiter/ops/aiter_operator.py`  (+63/-10)
- `aiter/ops/attention.py`  (+55/-4)
- `aiter/ops/gradlib.py`  (+43/-9)
- `aiter/ops/rmsnorm.py`  (+29/-18)
- `aiter/ops/batched_gemm_op_a8w8.py`  (+38/-6)
- `aiter/ops/norm.py`  (+31/-9)
- `aiter/ops/rope.py`  (+16/-16)
- `aiter/ops/topk.py`  (+24/-3)
- `aiter/ops/quant.py`  (+9/-7)
- `csrc/include/rocm_ops.hpp`  (+11/-5)

## Key added lines (kernel files)

**`aiter/jit/core.py`**
```
MANUAL_SCHEMA_OPS = [
"register_graph_buffers",
"module_moe_ck2stages",
"mha_fwd",
```

**`aiter/ops/activation.py`**
```
def silu_and_mul(out: Tensor, input: Tensor) -> None: ...
def scaled_silu_and_mul(out: Tensor, input: Tensor, scale: Tensor) -> None: ...
def gelu_and_mul(out: Tensor, input: Tensor) -> None: ...
def gelu_tanh_and_mul(out: Tensor, input: Tensor) -> None: ...
```

**`aiter/ops/aiter_operator.py`**
```
import torch
def binary_fake_shape(input: Tensor, other: Tensor) -> Tensor:
shape1 = list(input.shape)
shape2 = list(other.shape)
```

**`aiter/ops/attention.py`**
```
def gen_pa_fwd_native_fake(
query: torch.Tensor,
key_cache: torch.Tensor,
value_cache: torch.Tensor,
```

**`aiter/ops/batched_gemm_op_a8w8.py`**
```
def gen_batched_gemm_a8w8_fake_tensors(
XQ: Tensor,
WQ: Tensor,
x_scale: Tensor,
```
