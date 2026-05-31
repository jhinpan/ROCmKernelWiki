# Diff summary

- **files changed:** 24
- **lines:** +710 / -99
- **kernel-ish files:** 24

## Files (by churn)

- `aiter/ops/triton/_triton_kernels/moe_op_silu_fused.py`  (+90/-8)
- `aiter/ops/triton/_triton_kernels/moe_op.py`  (+88/-7)
- `aiter/ops/triton/_triton_kernels/moe_op_e2e.py`  (+47/-4)
- `aiter/ops/triton/extend_attention.py`  (+41/-9)
- `aiter/ops/triton/_triton_kernels/moe_op_gelu.py`  (+45/-4)
- `aiter/ops/triton/_triton_kernels/moe_align_block_size.py`  (+42/-6)
- `aiter/ops/triton/_triton_kernels/topk.py`  (+33/-4)
- `aiter/ops/triton/_triton_kernels/extend_attention.py`  (+27/-8)
- `aiter/ops/triton/moe_op.py`  (+27/-5)
- `aiter/ops/triton/moe_op_silu_fused.py`  (+26/-5)
- `aiter/ops/triton/moe_op_e2e.py`  (+25/-4)
- `aiter/ops/triton/moe_op_gelu.py`  (+24/-3)
- `aiter/ops/triton/moe_op_mxfp4.py`  (+25/-2)
- `aiter/ops/triton/moe_op_mxfp4_silu_fused.py`  (+25/-2)
- `aiter/ops/triton/_triton_kernels/moe_op_mxfp4.py`  (+22/-2)

## Key added lines (kernel files)

**`aiter/ops/triton/_triton_kernels/extend_attention.py`**
```
from ..utils._triton.pid_preprocessing import remap_xcd
from ..utils._triton.kernel_repr import make_kernel_repr
_fwd_kernel_extend_repr = make_kernel_repr(
"_fwd_kernel",
```

**`aiter/ops/triton/_triton_kernels/moe_align_block_size.py`**
```
from ..utils._triton.kernel_repr import make_kernel_repr
_moe_align_block_size_stage1_repr = make_kernel_repr(
"_moe_align_block_size_stage1_kernel",
"num_experts",
```

**`aiter/ops/triton/_triton_kernels/moe_op.py`**
```
from ..utils._triton.kernel_repr import make_kernel_repr
_fused_moe_kernel_gptq_awq_repr = make_kernel_repr(
"_fused_moe_kernel_gptq_awq",
"group_size",
```

**`aiter/ops/triton/_triton_kernels/moe_op_e2e.py`**
```
from ..utils._triton.kernel_repr import make_kernel_repr
_e2e_moe_kernel_repr = make_kernel_repr(
"e2e_moe_kernel",
"EVEN_K",
```

**`aiter/ops/triton/_triton_kernels/moe_op_gelu.py`**
```
from ..utils._triton.kernel_repr import make_kernel_repr
_fused_moe_kernel_gelu_repr = make_kernel_repr(
"_fused_moe_kernel",
"BLOCK_SCALE",
```
