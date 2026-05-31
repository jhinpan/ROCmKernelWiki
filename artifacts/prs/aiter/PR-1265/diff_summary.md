# Diff summary

- **files changed:** 13
- **lines:** +1039 / -2
- **kernel-ish files:** 11

## Files (by churn)

- `csrc/ck_deepgemm/gen_instances.py`  (+305/-0)
- `csrc/ck_deepgemm/include/deepgemm_common.cuh`  (+214/-0)
- `op_tests/test_deepgemm.py`  (+211/-0)
- `csrc/ck_deepgemm/deepgemm.cu`  (+149/-0)
- `csrc/ck_deepgemm/deepgemm_common.py`  (+64/-0)
- `aiter/ops/deepgemm.py`  (+30/-0)
- `csrc/ck_deepgemm/include/deepgemm.h`  (+24/-0)
- `aiter/jit/optCompilerConfig.json`  (+17/-0)
- `csrc/include/rocm_ops.hpp`  (+11/-0)
- `csrc/pybind/deepgemm_pybind.cu`  (+9/-0)
- `csrc/rocm_ops.cpp`  (+3/-1)
- `3rdparty/composable_kernel`  (+1/-1)
- `aiter/__init__.py`  (+1/-0)

## Key added lines (kernel files)

**`aiter/__init__.py`**
```
from .ops.deepgemm import *
```

**`aiter/ops/deepgemm.py`**
```
from torch import Tensor
from typing import Optional
from ..jit.core import (
compile_ops,
```

**`csrc/ck_deepgemm/deepgemm.cu`**
```
using RowwiseKernel = std::function<
torch::Tensor(torch::Tensor &, torch::Tensor &,
torch::Tensor &, torch::Tensor &,
std::optional<torch::Tensor>, std::optional<torch::Tensor>)>;
```

**`csrc/ck_deepgemm/deepgemm_common.py`**
```
from dataclasses import dataclass
@dataclass
class kernelInstance:
BLOCK_SIZE: int
```

**`csrc/ck_deepgemm/gen_instances.py`**
```
import os
from pathlib import Path
import pandas as pd
import argparse
```
