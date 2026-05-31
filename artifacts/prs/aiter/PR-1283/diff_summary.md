# Diff summary

- **files changed:** 18
- **lines:** +415 / -148
- **kernel-ish files:** 18

## Files (by churn)

- `aiter/ops/triton/gemm_afp4wfp4.py`  (+46/-36)
- `aiter/ops/triton/_triton_kernels/gemm_afp4wfp4.py`  (+67/-7)
- `aiter/ops/triton/_triton_kernels/gemm_a8wfp4.py`  (+38/-3)
- `aiter/ops/triton/gemm_a8wfp4.py`  (+19/-21)
- `aiter/ops/triton/_triton_kernels/gemm_a8w8_blockscale.py`  (+32/-2)
- `aiter/ops/triton/_triton_kernels/gemm_a8w8_per_token_scale.py`  (+30/-2)
- `aiter/ops/triton/gemm_a16w16.py`  (+16/-11)
- `aiter/ops/triton/gemm_a8w8.py`  (+14/-13)
- `aiter/ops/triton/gemm_a8w8_blockscale.py`  (+15/-11)
- `aiter/ops/triton/gemm_afp4wfp4_pre_quant_atomic.py`  (+14/-10)
- `aiter/ops/triton/_triton_kernels/gemm_afp4wfp4_pre_quant_atomic.py`  (+20/-2)
- `aiter/ops/triton/gemm_a16w16_gated.py`  (+12/-10)
- `aiter/ops/triton/gemm_a16w16_atomic.py`  (+12/-8)
- `aiter/ops/triton/gemm_a8w8_per_token_scale.py`  (+12/-8)
- `aiter/ops/triton/_triton_kernels/gemm_a16w16_atomic.py`  (+18/-1)

## Key added lines (kernel files)

**`aiter/ops/triton/_triton_kernels/gemm_a16w16.py`**
```
_gemm_a16w16_reduce_repr = make_kernel_repr(
"_gemm_a16w16_reduce_kernel",
"BLOCK_SIZE_M",
"BLOCK_SIZE_N",
```

**`aiter/ops/triton/_triton_kernels/gemm_a16w16_atomic.py`**
```
from ..utils._triton.kernel_repr import make_kernel_repr
_gemm_a16w16_atomic_repr = make_kernel_repr(
"_gemm_a16_w16_atomic_kernel",
"BLOCK_SIZE_M",
```

**`aiter/ops/triton/_triton_kernels/gemm_a16w16_gated.py`**
```
from ..utils._triton.kernel_repr import make_kernel_repr
_gemm_a16w16_gated_repr = make_kernel_repr(
"_gemm_a16_w16_gated_kernel",
"BLOCK_SIZE_M",
```

**`aiter/ops/triton/_triton_kernels/gemm_a8w8.py`**
```
from ..utils._triton.kernel_repr import make_kernel_repr
_gemm_a8w8_repr = make_kernel_repr(
"_gemm_a8w8_kernel",
"HAS_BIAS",
```

**`aiter/ops/triton/_triton_kernels/gemm_a8w8_blockscale.py`**
```
from ..utils._triton.kernel_repr import make_kernel_repr
_gemm_a8w8_blockscale_repr = make_kernel_repr(
"_gemm_a8w8_blockscale_kernel",
"GROUP_K",
```
