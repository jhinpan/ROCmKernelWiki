# Diff summary

- **files changed:** 13
- **lines:** +1102 / -24
- **kernel-ish files:** 10

## Files (by churn)

- `python/perf-kernels/streamk/utils/tuned.yaml`  (+343/-0)
- `python/perf-kernels/streamk/streamk_kernel_atomic.py`  (+158/-0)
- `python/perf-kernels/streamk/utils/grid_model.py`  (+134/-0)
- `python/perf-kernels/streamk/utils/gemm_wrapper.py`  (+118/-0)
- `python/perf-kernels/streamk/gemm_benchmark.py`  (+111/-0)
- `python/perf-kernels/streamk/persistent_gemm.py`  (+96/-0)
- `python/perf-kernels/streamk/utils/solution_selection.py`  (+54/-0)
- `python/perf-kernels/streamk/streamk_kernel.py`  (+25/-16)
- `python/perf-kernels/streamk/README.md`  (+40/-0)
- `python/perf-kernels/streamk/utils/file_generator.py`  (+11/-3)
- `python/perf-kernels/streamk/03-matrix-multiplication-stream-k.py`  (+5/-2)
- `python/perf-kernels/streamk/tune_streamk.py`  (+5/-2)
- `python/perf-kernels/streamk/utils/unittest.sh`  (+2/-1)

## Key added lines (kernel files)

**`python/perf-kernels/streamk/03-matrix-multiplication-stream-k.py`**
```
STREAMK_TILES=total_tiles_streamk,
m, n, k = 8192, 8192, 8192  # some problem size to test
```

**`python/perf-kernels/streamk/gemm_benchmark.py`**
```
import os
import json
import torch
import triton
```

**`python/perf-kernels/streamk/persistent_gemm.py`**
```
import triton
import triton.language as tl
@triton.jit()
def streamk_gemm(
```

**`python/perf-kernels/streamk/streamk_kernel.py`**
```
STREAMK_TILES: tl.constexpr,
total_full_tiles = total_tiles - STREAMK_TILES
tl.assume(stride_am > 0)
tl.assume(stride_ak > 0)
```

**`python/perf-kernels/streamk/streamk_kernel_atomic.py`**
```
import triton
import triton.language as tl
@triton.jit()
def streamk_gemm(
```
