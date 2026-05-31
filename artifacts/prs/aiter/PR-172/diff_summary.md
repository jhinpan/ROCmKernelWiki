# Diff summary

- **files changed:** 16
- **lines:** +1555 / -1
- **kernel-ish files:** 12

## Files (by churn)

- `csrc/ck_batched_gemm_a8w8/include/batched_gemm_a8w8_common.cuh`  (+314/-0)
- `csrc/ck_batched_gemm_a8w8/gen_instances.py`  (+304/-0)
- `csrc/ck_batched_gemm_a8w8/batched_gemm_a8w8.cu`  (+193/-0)
- `csrc/ck_batched_gemm_a8w8/batched_gemm_a8w8_tune.py`  (+172/-0)
- `csrc/ck_batched_gemm_a8w8/batched_gemm_a8w8_common.py`  (+152/-0)
- `csrc/ck_batched_gemm_a8w8/batched_gemm_a8w8_tune.cu`  (+103/-0)
- `aiter/ops/batched_gemm_op_a8w8.py`  (+97/-0)
- `op_tests/test_batched_gemm_a8w8.py`  (+78/-0)
- `aiter/jit/optCompilerConfig.json`  (+29/-1)
- `aiter/configs/a8w8_untuned_batched_gemm.csv`  (+27/-0)
- `aiter/configs/a8w8_tuned_batched_gemm.csv`  (+26/-0)
- `csrc/ck_batched_gemm_a8w8/include/batched_gemm_a8w8.h`  (+22/-0)
- `csrc/ck_batched_gemm_a8w8/README.md`  (+17/-0)
- `csrc/pybind/batched_gemm_a8w8_pybind.cu`  (+10/-0)
- `csrc/pybind/batched_gemm_a8w8_tune_pybind.cu`  (+10/-0)

## Key added lines (kernel files)

**`aiter/__init__.py`**
```
from .ops.batched_gemm_op_a8w8 import *
```

**`aiter/ops/batched_gemm_op_a8w8.py`**
```
import torch
from torch import Tensor
from typing import List, Optional
import functools
```

**`csrc/ck_batched_gemm_a8w8/batched_gemm_a8w8.cu`**
```
using BatchedRowwiseKernel = std::function<
torch::Tensor(torch::Tensor &, torch::Tensor &,
torch::Tensor &, torch::Tensor &,
torch::Tensor &, std::optional<torch::Tensor>,
```

**`csrc/ck_batched_gemm_a8w8/batched_gemm_a8w8_common.py`**
```
from dataclasses import dataclass
@dataclass
class kernelInstance:
BLOCK_SIZE: int
```

**`csrc/ck_batched_gemm_a8w8/batched_gemm_a8w8_tune.cu`**
```
using BatchedRowwiseKernel = std::function<
torch::Tensor(torch::Tensor &, torch::Tensor &,
torch::Tensor &, torch::Tensor &,
torch::Tensor &, std::optional<torch::Tensor>,
```
