# Diff summary

- **files changed:** 7
- **lines:** +696 / -220
- **kernel-ish files:** 6

## Files (by churn)

- `aiter/ops/triton/_gluon_kernels/gemm/basic/gemm_mxfp4.py`  (+388/-0)
- `op_tests/triton_tests/gemm/basic/test_gemm_afp4wfp4.py`  (+137/-71)
- `aiter/ops/triton/gemm/basic/gemm_afp4wfp4.py`  (+95/-46)
- `aiter/ops/triton/configs/gemm/gfx1250-GEMM-AFP4WFP4_PRESHUFFLED.json`  (+14/-84)
- `aiter/ops/shuffle.py`  (+38/-0)
- `op_tests/op_benchmarks/triton/bench_gemm_afp4wfp4.py`  (+22/-15)
- `aiter/ops/triton/gluon/gemm_afp4wfp4.py`  (+2/-4)

## Key added lines (kernel files)

**`aiter/ops/shuffle.py`**
```
def shuffle_weight_gfx1250(w: torch.Tensor) -> torch.Tensor:
Preshuffle weights for gfx1250 WMMA.
For 2D input (N, K): view as (N//16, 16, K//32, 2, 16) ->
permute(0, 2, 3, 1, 4) -> reshape (N//16, K*16).
```

**`aiter/ops/triton/_gluon_kernels/gemm/basic/gemm_mxfp4.py`**
```
from triton.experimental import gluon
import triton.experimental.gluon.language as gl
from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr
SCALE_GROUP_ELEMS = 32
```

**`aiter/ops/triton/gemm/basic/gemm_afp4wfp4.py`**
```
_gemm_afp4wfp4_kernel as _triton_gemm_afp4wfp4_kernel,
_gemm_afp4wfp4_preshuffle_kernel as _triton_gemm_afp4wfp4_preshuffle_kernel,
_gemm_afp4wfp4_kernel_preshuffle_scales as _triton_gemm_afp4wfp4_kernel_preshuffle_scales,
from aiter.ops.triton._gluon_kernels.gemm.basic.gemm_mxfp4 import (
```

**`aiter/ops/triton/gluon/gemm_afp4wfp4.py`**
```
if dev not in ["gfx950", "gfx1250"]:
raise ValueError("Gluon implementation is not supported on this device.")
```

**`op_tests/op_benchmarks/triton/bench_gemm_afp4wfp4.py`**
```
gemm_afp4wfp4_preshuffle,
metric: str,
layout: str,
preshuffle: bool,
```
