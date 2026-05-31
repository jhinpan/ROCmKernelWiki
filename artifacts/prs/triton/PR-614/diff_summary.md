# Diff summary

- **files changed:** 10
- **lines:** +2691 / -0
- **kernel-ish files:** 8

## Files (by churn)

- `python/perf-kernels/tune_gemm/tune_gemm.py`  (+937/-0)
- `python/perf-kernels/tune_gemm/matmul.py`  (+375/-0)
- `python/perf-kernels/tune_gemm/utils/file_generator.py`  (+355/-0)
- `python/perf-kernels/tune_gemm/rocprof_gemm.py`  (+318/-0)
- `python/perf-kernels/tune_gemm/README.md`  (+316/-0)
- `python/perf-kernels/tune_gemm/utils/utils.py`  (+115/-0)
- `python/perf-kernels/tune_gemm/icache_flush.py`  (+94/-0)
- `python/perf-kernels/tune_gemm/one_config.py`  (+90/-0)
- `python/perf-kernels/tune_gemm/matmul_kernel.py`  (+64/-0)
- `python/perf-kernels/tune_gemm/tune_gemm.sh`  (+27/-0)

## Key added lines (kernel files)

**`python/perf-kernels/tune_gemm/icache_flush.py`**
```
import ctypes
import array
import random
import math
```

**`python/perf-kernels/tune_gemm/matmul.py`**
```
Matrix Multiplication Tuning Scripts, Changed from the tutorial example "python/tutorials/03-matrix-multiplication.py"
import torch
import triton
import triton.language as tl
```

**`python/perf-kernels/tune_gemm/matmul_kernel.py`**
```
import triton
import triton.language as tl
@triton.jit
def matmul_kernel(
```

**`python/perf-kernels/tune_gemm/one_config.py`**
```
Script for running one Matrix Multiplication kernel config at a time
import argparse
import re
import sys
```

**`python/perf-kernels/tune_gemm/rocprof_gemm.py`**
```
import argparse
import sys
import torch
import triton
```
