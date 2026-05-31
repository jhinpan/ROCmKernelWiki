# Diff summary

- **files changed:** 17
- **lines:** +1003 / -96
- **kernel-ish files:** 5

## Files (by churn)

- `aiter/ops/triton/gemm_a8w8_per_token_scale.py`  (+363/-0)
- `aiter/ops/triton/gemm_a8w8_blockscale.py`  (+205/-92)
- `op_tests/op_benchmarks/triton/bench_gemm_a8w8_per_token_scale.py`  (+147/-0)
- `op_tests/triton_tests/test_gemm_a8w8_per_token_scale.py`  (+134/-0)
- `aiter/ops/triton/configs/gemm/MI300X-GEMM-A8W8_BLOCKSCALE.json`  (+15/-0)
- `aiter/ops/triton/configs/gemm/MI350X-GEMM-A8W8_BLOCKSCALE-N=1024-K=8192.json`  (+14/-0)
- `aiter/ops/triton/configs/gemm/MI350X-GEMM-A8W8_BLOCKSCALE-N=32768-K=8192.json`  (+14/-0)
- `aiter/ops/triton/configs/gemm/MI350X-GEMM-A8W8_BLOCKSCALE-N=8192-K=1024.json`  (+14/-0)
- `aiter/ops/triton/configs/gemm/MI350X-GEMM-A8W8_BLOCKSCALE-N=8192-K=32768.json`  (+14/-0)
- `aiter/ops/triton/configs/gemm/MI350X-GEMM-A8W8_BLOCKSCALE.json`  (+14/-0)
- `aiter/ops/triton/configs/gemm/MI350X-GEMM-A8W8_PER_TOKEN_SCALE-N=1024-K=8192.json`  (+14/-0)
- `aiter/ops/triton/configs/gemm/MI350X-GEMM-A8W8_PER_TOKEN_SCALE-N=32768-K=8192.json`  (+14/-0)
- `aiter/ops/triton/configs/gemm/MI350X-GEMM-A8W8_PER_TOKEN_SCALE-N=8192-K=1024.json`  (+14/-0)
- `aiter/ops/triton/configs/gemm/MI350X-GEMM-A8W8_PER_TOKEN_SCALE-N=8192-K=32768.json`  (+14/-0)
- `op_tests/triton_tests/test_gemm_a8w8_blockscale.py`  (+7/-1)

## Key added lines (kernel files)

**`aiter/ops/triton/gemm_a8w8_blockscale.py`**
```
import os
from aiter.ops.triton.utils.pid_preprocessing import pid_grid, remap_xcd
stride_ck,
NUM_KSPLIT: tl.constexpr,
```

**`aiter/ops/triton/gemm_a8w8_per_token_scale.py`**
```
from typing import Optional
import functools
import json
import os
```

**`op_tests/op_benchmarks/triton/bench_gemm_a8w8_per_token_scale.py`**
```
import sys
import torch
import triton
from aiter.ops.triton.gemm_a8w8_per_token_scale import gemm_a8w8_per_token_scale
```

**`op_tests/triton_tests/test_gemm_a8w8_blockscale.py`**
```
x_vals += [
(256, 8192, 1024),
(256, 1024, 8192),
(256, 32768, 8192),
```

**`op_tests/triton_tests/test_gemm_a8w8_per_token_scale.py`**
```
import torch
import triton
import pytest
from aiter.ops.triton.gemm_a8w8_per_token_scale import gemm_a8w8_per_token_scale
```
