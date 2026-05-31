# Diff summary

- **files changed:** 17
- **lines:** +1319 / -1
- **kernel-ish files:** 12

## Files (by churn)

- `csrc/ck_batched_gemm_bf16/gen_instances.py`  (+265/-0)
- `csrc/ck_batched_gemm_bf16/include/batched_gemm_bf16_common.cuh`  (+191/-0)
- `csrc/ck_batched_gemm_bf16/batched_gemm_bf16_tune.py`  (+168/-0)
- `csrc/ck_batched_gemm_bf16/batched_gemm_bf16.cu`  (+162/-0)
- `csrc/ck_batched_gemm_bf16/batched_gemm_bf16_common.py`  (+152/-0)
- `aiter/ops/batched_gemm_op_bf16.py`  (+91/-0)
- `csrc/ck_batched_gemm_bf16/batched_gemm_bf16_tune.cu`  (+79/-0)
- `op_tests/test_batched_gemm_bf16.py`  (+74/-0)
- `aiter/jit/optCompilerConfig.json`  (+28/-0)
- `aiter/configs/bf16_tuned_batched_gemm.csv`  (+27/-0)
- `aiter/configs/bf16_untuned_batched_gemm.csv`  (+27/-0)
- `csrc/ck_batched_gemm_bf16/include/batched_gemm_bf16.h`  (+18/-0)
- `csrc/ck_batched_gemm_bf16/README.md`  (+17/-0)
- `csrc/pybind/batched_gemm_bf16_pybind.cu`  (+9/-0)
- `csrc/pybind/batched_gemm_bf16_tune_pybind.cu`  (+9/-0)

## Key added lines (kernel files)

**`aiter/__init__.py`**
```
from .ops.batched_gemm_op_bf16 import *
```

**`aiter/ops/batched_gemm_op_bf16.py`**
```
import torch
from torch import Tensor
from typing import List, Optional
import functools
```

**`csrc/ck_batched_gemm_bf16/batched_gemm_bf16.cu`**
```
using BatchedKernel = std::function<
torch::Tensor(torch::Tensor &, torch::Tensor &,
torch::Tensor &, std::optional<torch::Tensor>,
struct IntTupleHash
```

**`csrc/ck_batched_gemm_bf16/batched_gemm_bf16_common.py`**
```
from dataclasses import dataclass
@dataclass
class kernelInstance:
BLOCK_SIZE: int
```

**`csrc/ck_batched_gemm_bf16/batched_gemm_bf16_tune.cu`**
```
using BatchedKernel = std::function<
torch::Tensor(torch::Tensor &, torch::Tensor &,
torch::Tensor &, std::optional<torch::Tensor>,
using BatchedKernelMap = std::unordered_map<
```
