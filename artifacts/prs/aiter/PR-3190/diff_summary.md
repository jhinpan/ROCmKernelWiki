# Diff summary

- **files changed:** 12
- **lines:** +1920 / -8
- **kernel-ish files:** 7

## Files (by churn)

- `aiter/ops/triton/_triton_kernels/fusions/mhc.py`  (+491/-0)
- `aiter/ops/triton/fusions/mhc.py`  (+366/-0)
- `aiter/ops/triton/_triton_kernels/attention/pa_decode_sparse.py`  (+304/-0)
- `aiter/ops/triton/attention/pa_decode_sparse.py`  (+192/-0)
- `op_tests/triton_tests/attention/test_pa_decode_sparse.py`  (+163/-0)
- `op_tests/triton_tests/fusions/test_mhc.py`  (+127/-5)
- `aiter/ops/triton/configs/gemm/gfx950-GEMM-A8W8_BLOCKSCALE_PRESHUFFLED-N=1536-K=4096.json`  (+98/-0)
- `aiter/ops/triton/configs/gemm/gfx950-GEMM-A8W8_BLOCKSCALE_PRESHUFFLED-N=2048-K=7168.json`  (+98/-0)
- `aiter/ops/triton/configs/gfx950-MHC_FUSED_SINKHORN-C=7168.json`  (+65/-0)
- `aiter/ops/triton/configs/gfx950-MHC_FUSED_SINKHORN-C=4096.json`  (+9/-0)
- `aiter/ops/triton/configs/gfx950-MHC_POST.json`  (+3/-3)
- `aiter/ops/triton/_triton_kernels/fusions/__init__.py`  (+4/-0)

## Key added lines (kernel files)

**`aiter/ops/triton/_triton_kernels/attention/pa_decode_sparse.py`**
```
"""Sparse paged-decode attention kernels (split-K + per-token paged indices).
Two-kernel decomposition of a flash-decode whose K range for each token is a
gathered subset of a unified KV pool:
``_pa_decode_sparse``        : split-K main kernel. Grid (T, ceil(H/BLOCK_H),
```

**`aiter/ops/triton/_triton_kernels/fusions/__init__.py`**
```
_mhc_post_pre_split_kernel,
_mhc_post_pre_reduce_apply_kernel,
"_mhc_post_pre_split_kernel",
"_mhc_post_pre_reduce_apply_kernel",
```

**`aiter/ops/triton/_triton_kernels/fusions/mhc.py`**
```
from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr
_mhc_post_pre_split_kernel_repr = make_kernel_repr(
"_mhc_post_pre_split_kernel",
"stride_phi_k",
```

**`aiter/ops/triton/attention/pa_decode_sparse.py`**
```
"""Sparse paged-decode attention over a unified KV pool with per-token paged
indices. See ``_triton_kernels/attention/pa_decode_sparse.py`` for the
kernels' caller contract.
This module exposes ``pa_decode_sparse`` — a 3D split-K + widened-BLOCK_H
```

**`aiter/ops/triton/fusions/mhc.py`**
```
_mhc_post_pre_split_kernel,
_mhc_post_pre_reduce_apply_kernel,
def mhc_post_pre(
layer_input: torch.Tensor,  # (M, C)        bf16 / fp16
```
