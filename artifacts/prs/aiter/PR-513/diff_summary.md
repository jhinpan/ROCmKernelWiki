# Diff summary

- **files changed:** 17
- **lines:** +2801 / -85
- **kernel-ish files:** 13

## Files (by churn)

- `csrc/ck_gemm_a8w8_bpreshuffle/gemm_a8w8_bpreshuffle_common.py`  (+1528/-0)
- `csrc/ck_gemm_a8w8_bpreshuffle/gen_instances.py`  (+292/-0)
- `csrc/ck_gemm_a8w8_bpreshuffle/include/gemm_a8w8_bpreshuffle_common.cuh`  (+212/-0)
- `csrc/ck_gemm_a8w8_bpreshuffle/gemm_a8w8_bpreshuffle_tune.py`  (+196/-0)
- `csrc/rocm_ops.cpp`  (+63/-62)
- `csrc/ck_gemm_a8w8_bpreshuffle/gemm_a8w8_bpreshuffle.cu`  (+122/-0)
- `csrc/ck_gemm_a8w8_bpreshuffle/gemm_a8w8_bpreshuffle_tune.cu`  (+86/-0)
- `op_tests/test_gemm_a8w8.py`  (+44/-21)
- `aiter/configs/a8w8_bpreshuffle_tuned_gemm.csv`  (+56/-0)
- `aiter/configs/a8w8_bpreshuffle_untuned_gemm.csv`  (+56/-0)
- `aiter/ops/gemm_op_a8w8.py`  (+36/-2)
- `aiter/jit/optCompilerConfig.json`  (+30/-0)
- `csrc/ck_gemm_a8w8_bpreshuffle/README.md`  (+30/-0)
- `csrc/include/rocm_ops.hpp`  (+20/-0)
- `csrc/ck_gemm_a8w8_bpreshuffle/include/gemm_a8w8_bpreshuffle.h`  (+18/-0)

## Key added lines (kernel files)

**`aiter/ops/gemm_op_a8w8.py`**
```
@compile_ops("module_gemm_a8w8_bpreshuffle", fc_name="gemm_a8w8_bpreshuffle")
def gemm_a8w8_bpreshuffle(
XQ: Tensor,
WQ: Tensor,
```

**`csrc/ck_gemm_a8w8_bpreshuffle/gemm_a8w8_bpreshuffle.cu`**
```
using RowwiseKernel = std::function<torch::Tensor(
torch::Tensor&, torch::Tensor&, torch::Tensor&, torch::Tensor&, torch::Tensor&)>;
struct IntTupleHash
size_t operator()(const std::tuple<int, int, int>& t) const
```

**`csrc/ck_gemm_a8w8_bpreshuffle/gemm_a8w8_bpreshuffle_common.py`**
```
from dataclasses import dataclass
@dataclass
class kernelInstance:
BLOCK_SIZE: int
```

**`csrc/ck_gemm_a8w8_bpreshuffle/gemm_a8w8_bpreshuffle_tune.cu`**
```
using BlockwiseKernel = std::function<torch::Tensor(
torch::Tensor&, torch::Tensor&, torch::Tensor&, torch::Tensor&, torch::Tensor&)>;
using BlockwiseKernelMap = std::unordered_map<int, BlockwiseKernel>;
static constexpr int nextPow2(unsigned int num)
```

**`csrc/ck_gemm_a8w8_bpreshuffle/gemm_a8w8_bpreshuffle_tune.py`**
```
import os
import aiter
import pandas as pd
import torch
```
