# Diff summary

- **files changed:** 12
- **lines:** +988 / -23
- **kernel-ish files:** 7

## Files (by churn)

- `aiter/ops/triton/gemm_a16w16_gated.py`  (+241/-0)
- `op_tests/op_benchmarks/triton/bench_gemm_a16w16_gating.py`  (+223/-0)
- `op_tests/triton_tests/test_gemm_a16w16_gated.py`  (+87/-0)
- `aiter/ops/triton/configs/gemm/MI300X-GEMM-A16W16-gated.json`  (+74/-0)
- `aiter/ops/triton/configs/gemm/MI350X-GEMM-A16W16-gated.json`  (+74/-0)
- `aiter/ops/triton/configs/gemm/MI350X-GEMM-A16W16-N=256-K=7168.json`  (+57/-9)
- `aiter/ops/triton/configs/gemm/MI350X-GEMM-A16W16.json`  (+63/-3)
- `aiter/ops/triton/configs/gemm/MI300X-GEMM-A16W16.json`  (+61/-1)
- `op_tests/triton_tests/test_gemm_a16w16.py`  (+56/-3)
- `op_tests/op_benchmarks/triton/bench_gemm_a16w16.py`  (+26/-3)
- `aiter/ops/triton/gemm_a16w16.py`  (+19/-4)
- `aiter/ops/triton/activation.py`  (+7/-0)

## Key added lines (kernel files)

**`aiter/ops/triton/activation.py`**
```
@triton.jit
def _relu(x):
return tl.maximum(0.0, x)
"silu_exp2": _silu_exp2,
```

**`aiter/ops/triton/gemm_a16w16.py`**
```
from aiter.ops.triton.activation import _get_activation_from_str
activation: tl.constexpr,
use_activation: tl.constexpr,
if use_activation:
```

**`aiter/ops/triton/gemm_a16w16_gated.py`**
```
from typing import Optional
import functools
import json
import os
```

**`op_tests/op_benchmarks/triton/bench_gemm_a16w16.py`**
```
from typing import Optional
metric: str,
layout: str,
atomic: bool = False,
```

**`op_tests/op_benchmarks/triton/bench_gemm_a16w16_gating.py`**
```
import sys
import torch
import triton
import math
```
