# Diff summary

- **files changed:** 10 (diff was byte-capped; summary is partial)
- **lines:** +1280 / -810
- **kernel-ish files:** 8

## Files (by churn)

- `aiter/configs/a8w8_bpreshuffle_tuned_gemm.csv`  (+523/-460)
- `aiter/ops/flydsl/kernels/moe_gemm_2stage.py`  (+206/-270)
- `aiter/ops/flydsl/gemm_tune/flydsl_gemm_a8w8_bpreshuffle_common.py`  (+335/-0)
- `aiter/ops/flydsl/gemm_kernels.py`  (+150/-0)
- `aiter/ops/flydsl/kernels/mfma_preshuffle_pipeline.py`  (+40/-32)
- `aiter/configs/model_configs/a8w8_bpreshuffle_tuned_gemm_dsv3.csv`  (+0/-24)
- `aiter/ops/flydsl/kernels/mfma_epilogues.py`  (+5/-14)
- `aiter/ops/flydsl/kernels/mixed_moe_gemm_2stage.py`  (+6/-10)
- `aiter/ops/flydsl/kernels/kernels_common.py`  (+11/-0)
- `aiter/ops/flydsl/__init__.py`  (+4/-0)

## Key added lines (kernel files)

**`aiter/ops/flydsl/__init__.py`**
```
from .gemm_kernels import (
flydsl_preshuffle_gemm_a8,
"flydsl_preshuffle_gemm_a8",
```

**`aiter/ops/flydsl/gemm_kernels.py`**
```
from torch import Tensor
from aiter import logger
from aiter.utility import dtypes
from .utils import is_flydsl_available
```

**`aiter/ops/flydsl/gemm_tune/flydsl_gemm_a8w8_bpreshuffle_common.py`**
```
from dataclasses import dataclass
import math
import os
def get_gfx():
```

**`aiter/ops/flydsl/kernels/kernels_common.py`**
```
from flydsl.runtime.device import get_rocm_arch, is_rdna_arch
def get_warp_size(arch=None):
"""Return the wavefront/warp size for the given GPU architecture.
CDNA (gfx9xx) uses wave64, RDNA (gfx10xx/gfx11xx/gfx12xx) uses wave32.
```

**`aiter/ops/flydsl/kernels/mfma_epilogues.py`**
```
import flydsl.expr as fx
ii_idx_list = [fx.Index(ii) for ii in range(4)]
c_nlane = fx.Index(CShuffleNLane)
m_lane = tx // c_nlane
```
