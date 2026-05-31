# Diff summary

- **files changed:** 10
- **lines:** +183 / -76
- **kernel-ish files:** 10

## Files (by churn)

- `aiter/ops/triton/_triton_kernels/batched_gemm_afp4wfp4_pre_quant.py`  (+32/-3)
- `aiter/ops/triton/batched_gemm_a8w8.py`  (+15/-17)
- `aiter/ops/triton/batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant.py`  (+17/-15)
- `aiter/ops/triton/_triton_kernels/batched_gemm_afp4wfp4.py`  (+29/-2)
- `aiter/ops/triton/batched_gemm_afp4wfp4.py`  (+14/-13)
- `aiter/ops/triton/batched_gemm_afp4wfp4_pre_quant.py`  (+14/-13)
- `aiter/ops/triton/_triton_kernels/batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant.py`  (+19/-1)
- `aiter/ops/triton/batched_gemm_bf16.py`  (+11/-8)
- `aiter/ops/triton/_triton_kernels/batched_gemm_a8w8.py`  (+16/-2)
- `aiter/ops/triton/_triton_kernels/batched_gemm_bf16.py`  (+16/-2)

## Key added lines (kernel files)

**`aiter/ops/triton/_triton_kernels/batched_gemm_a8w8.py`**
```
from ..utils._triton.kernel_repr import make_kernel_repr
_batched_gemm_a8w8_repr = make_kernel_repr(
"_batched_gemm_a8w8_kernel",
"HAS_BIAS",
```

**`aiter/ops/triton/_triton_kernels/batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant.py`**
```
from ..utils._triton.kernel_repr import make_kernel_repr
_batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant_repr = make_kernel_repr(
"_batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant_kernel",
"HAS_BIAS",
```

**`aiter/ops/triton/_triton_kernels/batched_gemm_afp4wfp4.py`**
```
from ..utils._triton.kernel_repr import make_kernel_repr
_batched_gemm_afp4_wfp4_repr = make_kernel_repr(
"_batched_gemm_afp4_wfp4_kernel",
"BLOCK_SIZE_M",
```

**`aiter/ops/triton/_triton_kernels/batched_gemm_afp4wfp4_pre_quant.py`**
```
from ..utils._triton.kernel_repr import make_kernel_repr
_batched_gemm_afp4_wfp4_pre_quant_repr = make_kernel_repr(
"_batched_gemm_afp4_wfp4_pre_quant_kernel",
"BLOCK_SIZE_M",
```

**`aiter/ops/triton/_triton_kernels/batched_gemm_bf16.py`**
```
from ..utils._triton.kernel_repr import make_kernel_repr
_batched_gemm_bf16_repr = make_kernel_repr(
"_batched_gemm_bf16_kernel",
"HAS_BIAS",
```
