# Diff summary

- **files changed:** 8
- **lines:** +1909 / -131
- **kernel-ish files:** 6

## Files (by churn)

- `aiter/ops/flydsl/kernels/splitk_hgemm.py`  (+763/-0)
- `aiter/ops/flydsl/gemm_kernels.py`  (+472/-0)
- `aiter/ops/flydsl/kernels/tensor_shim.py`  (+275/-0)
- `aiter/configs/model_configs/kimik2_bf16_tuned_gemm.csv`  (+125/-125)
- `aiter/ops/flydsl/test_flydsl_splitk_hgemm.py`  (+210/-0)
- `aiter/tuned_gemm.py`  (+61/-1)
- `aiter/configs/model_configs/dsv3_bf16_tuned_gemm.csv`  (+0/-5)
- `aiter/ops/flydsl/__init__.py`  (+3/-0)

## Key added lines (kernel files)

**`aiter/ops/flydsl/__init__.py`**
```
from .gemm_kernels import flydsl_hgemm
"flydsl_hgemm",
```

**`aiter/ops/flydsl/gemm_kernels.py`**
```
"""High-level FlyDSL HGEMM APIs."""
from __future__ import annotations
from itertools import product
from typing import Dict, Optional
```

**`aiter/ops/flydsl/kernels/splitk_hgemm.py`**
```
import functools
import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir import ir
```

**`aiter/ops/flydsl/kernels/tensor_shim.py`**
```
import numpy as np
from itertools import product
from abc import ABC, abstractmethod
from flydsl._mlir import ir
```

**`aiter/ops/flydsl/test_flydsl_splitk_hgemm.py`**
```
"""Unit tests for FlyDSL preshuffle split-K HGEMM kernels."""
from __future__ import annotations
import argparse
import pytest
```
