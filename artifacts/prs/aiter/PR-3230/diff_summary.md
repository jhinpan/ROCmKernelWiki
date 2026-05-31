# Diff summary

- **files changed:** 17
- **lines:** +203 / -432
- **kernel-ish files:** 17

## Files (by churn)

- `aiter/ops/triton/_triton_kernels/common/splitk_reduce.py`  (+89/-0)
- `aiter/ops/triton/_triton_kernels/gemm/basic/gemm_a16w16.py`  (+0/-81)
- `aiter/ops/triton/_triton_kernels/gemm/basic/gemm_a8w8_blockscale.py`  (+0/-67)
- `aiter/ops/triton/_triton_kernels/gemm/basic/gemm_a8w8_per_token_scale.py`  (+0/-67)
- `aiter/ops/triton/_triton_kernels/gemm/basic/gemm_a8w8.py`  (+0/-65)
- `aiter/ops/triton/_triton_kernels/gemm/basic/gemm_a8wfp4.py`  (+0/-63)
- `aiter/ops/triton/_triton_kernels/gemm/basic/gemm_afp4wfp4.py`  (+0/-59)
- `aiter/ops/triton/gemm/basic/gemm_afp4wfp4.py`  (+21/-4)
- `aiter/ops/triton/utils/_triton/kernel_repr.py`  (+17/-4)
- `aiter/ops/triton/gemm/basic/gemm_a16w8_blockscale.py`  (+14/-4)
- `aiter/ops/triton/gemm/basic/gemm_a16wfp4.py`  (+14/-4)
- `aiter/ops/triton/gemm/basic/gemm_a8w8_blockscale.py`  (+15/-3)
- `aiter/ops/triton/gemm/basic/gemm_a16w16.py`  (+7/-4)
- `aiter/ops/triton/gemm/basic/gemm_a8w8.py`  (+8/-3)
- `aiter/ops/triton/gemm/basic/gemm_a8w8_per_token_scale.py`  (+9/-2)

## Key added lines (kernel files)

**`aiter/ops/triton/_triton_kernels/common/splitk_reduce.py`**
```
import triton
import triton.language as tl
from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr
_gemm_splitk_reduce_repr = make_kernel_repr(
```

**`aiter/ops/triton/gemm/basic/gemm_a16w16.py`**
```
from aiter.ops.triton._triton_kernels.common.splitk_reduce import (
_gemm_splitk_reduce_kernel,
_gemm_splitk_reduce_kernel[grid_reduce](
ADD_BIAS=(bias is not None),
```

**`aiter/ops/triton/gemm/basic/gemm_a16w8_blockscale.py`**
```
from aiter.ops.triton._triton_kernels.common.splitk_reduce import (
_gemm_splitk_reduce_kernel,
_gemm_splitk_reduce_kernel[grid_reduce](
ADD_BIAS=False,
```

**`aiter/ops/triton/gemm/basic/gemm_a16wfp4.py`**
```
from aiter.ops.triton._triton_kernels.common.splitk_reduce import (
_gemm_splitk_reduce_kernel,
_gemm_splitk_reduce_kernel[grid_reduce](
ADD_BIAS=False,
```

**`aiter/ops/triton/gemm/basic/gemm_a8w8.py`**
```
from aiter.ops.triton._triton_kernels.common.splitk_reduce import (
_gemm_splitk_reduce_kernel,
_gemm_splitk_reduce_kernel[grid_reduce](
ADD_BIAS=bias is not None,
```
