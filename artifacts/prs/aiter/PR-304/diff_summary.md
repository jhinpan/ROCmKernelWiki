# Diff summary

- **files changed:** 10
- **lines:** +534 / -135
- **kernel-ish files:** 10

## Files (by churn)

- `aiter/ops/triton/gemm_a16w16.py`  (+151/-0)
- `aiter/ops/triton/quant.py`  (+39/-102)
- `op_benchmarks/triton/bench_gemm_a8w8_blockscale.py`  (+136/-0)
- `op_benchmarks/triton/bench_gemm_a16w16.py`  (+101/-0)
- `op_tests/triton/test_gemm_a16w16.py`  (+58/-0)
- `op_tests/triton/test_gemm_a8w8_blockscale.py`  (+22/-14)
- `op_tests/triton/test_quant.py`  (+12/-11)
- `op_tests/triton/test_moe.py`  (+6/-4)
- `aiter/ops/triton/gemm_a8w8_blockscale.py`  (+7/-2)
- `op_tests/triton/test_gemm_a8w8.py`  (+2/-2)

## Key added lines (kernel files)

**`aiter/ops/triton/gemm_a16w16.py`**
```
from typing import Optional
import torch
import triton
import triton.language as tl
```

**`aiter/ops/triton/gemm_a8w8_blockscale.py`**
```
def get_arch():
return triton.runtime.driver.active.get_current_target().arch
kpack = 1 if get_arch() in ('gfx950') else 2
num_warps = 4
```

**`aiter/ops/triton/quant.py`**
```
tl.assume(pid > 0)
tl.assume(x_in_stride_r > 0)
offs = pid * x_in_stride_r + tl.arange(0, NUM_COL_POW2)
mask = tl.arange(0, NUM_COL_POW2) < cols
```

**`op_benchmarks/triton/bench_gemm_a16w16.py`**
```
import argparse
import sys
import torch
import triton
```

**`op_benchmarks/triton/bench_gemm_a8w8_blockscale.py`**
```
import argparse
import sys
import torch
import triton
```
