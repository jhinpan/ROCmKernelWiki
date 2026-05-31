# Diff summary

- **files changed:** 13
- **lines:** +241 / -850
- **kernel-ish files:** 13

## Files (by churn)

- `aiter/ops/triton/_triton_kernels/moe/moe_op_gemm_a4w4.py`  (+3/-91)
- `aiter/ops/triton/_triton_kernels/moe/moe_op_gemm_a8w4.py`  (+3/-91)
- `aiter/ops/triton/_triton_kernels/moe/moe_op_gemm_a8w8.py`  (+3/-91)
- `aiter/ops/triton/_triton_kernels/moe/moe_op_gemm_a8w8_blockscale.py`  (+3/-91)
- `aiter/ops/triton/_triton_kernels/moe/moe_op_gemm_int8_smoothquant.py`  (+1/-93)
- `aiter/ops/triton/moe/moe_op_gemm_a4w4.py`  (+11/-78)
- `aiter/ops/triton/moe/moe_op_gemm_a8w4.py`  (+11/-78)
- `aiter/ops/triton/moe/moe_op_gemm_a8w8.py`  (+11/-78)
- `aiter/ops/triton/moe/moe_op_gemm_a8w8_blockscale.py`  (+11/-78)
- `aiter/ops/triton/moe/moe_op_gemm_int8_smoothquant.py`  (+4/-81)
- `aiter/ops/triton/_triton_kernels/moe/reduce.py`  (+75/-0)
- `aiter/ops/triton/moe/reduce.py`  (+73/-0)
- `aiter/ops/triton/_triton_kernels/moe/activations.py`  (+32/-0)

## Key added lines (kernel files)

**`aiter/ops/triton/_triton_kernels/moe/activations.py`**
```
import triton
import triton.language as tl
@triton.jit
def clip(x, limit, clip_lower: tl.constexpr):
```

**`aiter/ops/triton/_triton_kernels/moe/moe_op_gemm_a4w4.py`**
```
from aiter.ops.triton._triton_kernels.moe.activations import _swiglu
ADD_RESIDUAL: tl.constexpr,
out = _swiglu(acc, alpha, limit, ADD_RESIDUAL=ADD_RESIDUAL)
```

**`aiter/ops/triton/_triton_kernels/moe/moe_op_gemm_a8w4.py`**
```
from aiter.ops.triton._triton_kernels.moe.activations import _swiglu
ADD_RESIDUAL: tl.constexpr,
out = _swiglu(acc, alpha, limit, ADD_RESIDUAL=ADD_RESIDUAL)
```

**`aiter/ops/triton/_triton_kernels/moe/moe_op_gemm_a8w8.py`**
```
from aiter.ops.triton._triton_kernels.moe.activations import _swiglu
ADD_RESIDUAL: tl.constexpr,
out = _swiglu(acc, alpha, limit, ADD_RESIDUAL=ADD_RESIDUAL)
```

**`aiter/ops/triton/_triton_kernels/moe/moe_op_gemm_a8w8_blockscale.py`**
```
from aiter.ops.triton._triton_kernels.moe.activations import _swiglu
ADD_RESIDUAL: tl.constexpr,
out = _swiglu(acc, alpha, limit, ADD_RESIDUAL=ADD_RESIDUAL)
```
