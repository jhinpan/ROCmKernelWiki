# Diff summary

- **files changed:** 8
- **lines:** +2056 / -285
- **kernel-ish files:** 7

## Files (by churn)

- `aiter/ops/flydsl/kernels/small_m_hgemm.py`  (+1405/-0)
- `aiter/ops/flydsl/gemm_kernels.py`  (+323/-62)
- `aiter/ops/flydsl/kernels/splitk_hgemm.py`  (+106/-93)
- `aiter/configs/model_configs/gptoss_bf16_tuned_gemm.csv`  (+57/-57)
- `aiter/aot/flydsl/gemm.py`  (+42/-59)
- `aiter/ops/flydsl/kernels/hgemm_dispatch.py`  (+76/-0)
- `gradlib/gradlib/GemmTuner.py`  (+27/-12)
- `aiter/tuned_gemm.py`  (+20/-2)

## Key added lines (kernel files)

**`aiter/aot/flydsl/gemm.py`**
```
from aiter.ops.flydsl.gemm_kernels import get_flydsl_splitk_hgemm_kernel_params
from aiter.ops.flydsl.kernels.hgemm_dispatch import compile_flydsl_hgemm_kernel
def _parse_bool(value: Optional[str]) -> bool:
if value is None:
```

**`aiter/ops/flydsl/gemm_kernels.py`**
```
import re
from .kernels.hgemm_dispatch import compile_flydsl_hgemm_kernel
from .kernels.small_m_hgemm import iter_small_m_registry_configs
KERNEL_FAMILY_HGEMM = "hgemm"
```

**`aiter/ops/flydsl/kernels/hgemm_dispatch.py`**
```
from __future__ import annotations
from typing import Optional
from .small_m_hgemm import compile_small_m_hgemm_kernel
from .splitk_hgemm import compile_hgemm_kernel
```

**`aiter/ops/flydsl/kernels/small_m_hgemm.py`**
```
"""Dedicated small-M bf16 HGEMM kernel path.
This module intentionally stays separate from `hgemm.py`. The generic HGEMM
kernel and this small-M path share the same split-K contract and both still
take `m` as a runtime value, but this path is no longer just a different
```

**`aiter/ops/flydsl/kernels/splitk_hgemm.py`**
```
HAS_BIAS: bool = False,
if HAS_BIAS:
KERNEL_NAME += "_BIAS"
BIAS: fx.Tensor,
```
