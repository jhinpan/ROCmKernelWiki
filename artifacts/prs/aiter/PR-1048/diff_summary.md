# Diff summary

- **files changed:** 27
- **lines:** +701 / -48
- **kernel-ish files:** 12

## Files (by churn)

- `csrc/py_itfs_cu/asm_gemm_a16w16.cu`  (+287/-0)
- `op_tests/test_gemm.py`  (+128/-41)
- `hsa/gfx942/bf16gemm/codegen.py`  (+66/-0)
- `hsa/gfx950/bf16gemm/codegen.py`  (+66/-0)
- `aiter/ops/gemm_op_a16w16.py`  (+53/-0)
- `aiter/tuned_gemm.py`  (+26/-6)
- `aiter/jit/optCompilerConfig.json`  (+12/-0)
- `csrc/include/asm_gemm_a16w16.h`  (+11/-0)
- `csrc/include/rocm_ops.hpp`  (+11/-0)
- `aiter/configs/tuned_gemm.csv`  (+10/-0)
- `aiter/utility/dtypes.py`  (+8/-0)
- `csrc/pybind/gemm_a16w16_asm_pybind.cu`  (+6/-0)
- `hsa/gfx942/bf16gemm/bf16gemm_outf32.csv`  (+6/-0)
- `hsa/gfx950/bf16gemm/bf16gemm_outf32.csv`  (+6/-0)
- `.github/workflows/vllm_benchmark.yaml`  (+2/-1)

## Key added lines (kernel files)

**`aiter/__init__.py`**
```
from .ops.gemm_op_a16w16 import *
```

**`aiter/ops/gemm_op_a16w16.py`**
```
import torch
from torch import Tensor
from typing import Optional
from aiter import logger
```

**`aiter/tuned_gemm.py`**
```
from aiter import gemm_a16w16_asm
else s.kernelName
assert hipblasltKernelNames.equals(bestsols["kernelName"]), (
"""bf16/fp16 with per tensor fp8 quant"""
```

**`aiter/utility/dtypes.py`**
```
def str2Dtype(v):
parts = v.strip("()").split(",")
return list(d_dtypes[p.strip()] for p in parts)
except Exception as e:
```

**`csrc/include/asm_gemm_a16w16.h`**
```
torch::Tensor gemm_a16w16_asm(torch::Tensor& A,   // A:[M, K] bf16
torch::Tensor& B,   // B:[N, K] bf16
torch::Tensor& out, // Out:[M, N] f32
std::optional<torch::Tensor> bias,
```
