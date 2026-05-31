# Diff summary

- **files changed:** 18
- **lines:** +1467 / -4
- **kernel-ish files:** 13

## Files (by churn)

- `csrc/ck_gemm_a8w8_blockscale/gen_instances.py`  (+272/-0)
- `aiter/configs/a8w8_blockscale_untuned_gemm.csv`  (+234/-0)
- `csrc/ck_gemm_a8w8_blockscale/gemm_a8w8_blockscale_tune.py`  (+186/-0)
- `csrc/ck_gemm_a8w8_blockscale/include/gemm_a8w8_blockscale_common.cuh`  (+166/-0)
- `aiter/configs/a8w8_blockscale_tuned_gemm.csv`  (+118/-0)
- `csrc/ck_gemm_a8w8_blockscale/gemm_a8w8_blockscale.cu`  (+117/-0)
- `csrc/ck_gemm_a8w8_blockscale/gemm_a8w8_blockscale_common.py`  (+99/-0)
- `csrc/ck_gemm_a8w8_blockscale/gemm_a8w8_blockscale_tune.cu`  (+89/-0)
- `op_tests/test_gemm_a8w8_blockscale.py`  (+58/-0)
- `aiter/ops/gemm_op_a8w8.py`  (+38/-0)
- `aiter/jit/optCompilerConfig.json`  (+31/-0)
- `csrc/ck_gemm_a8w8_blockscale/include/gemm_a8w8_blockscale.h`  (+20/-0)
- `csrc/ck_gemm_a8w8_blockscale/README.md`  (+18/-0)
- `csrc/pybind/gemm_a8w8_blockscale_tune_pybind.cu`  (+10/-0)
- `csrc/pybind/gemm_a8w8_blockscale_pybind.cu`  (+9/-0)

## Key added lines (kernel files)

**`aiter/__init__.py`**
```
from .ops.mha import *
```

**`aiter/ops/gemm_op_a8w8.py`**
```
@compile_ops("module_gemm_a8w8_blockscale", fc_name="gemm_a8w8_blockscale")
def gemm_a8w8_blockscale(
XQ: Tensor,
WQ: Tensor,
```

**`csrc/ck_gemm_a8w8_blockscale/gemm_a8w8_blockscale.cu`**
```
using BlockwiseKernel = std::function<
torch::Tensor(torch::Tensor &, torch::Tensor &,
torch::Tensor &, torch::Tensor &,
torch::Tensor &)>;
```

**`csrc/ck_gemm_a8w8_blockscale/gemm_a8w8_blockscale_common.py`**
```
from dataclasses import dataclass
@dataclass
class kernelInstance:
BLOCK_SIZE: int
```

**`csrc/ck_gemm_a8w8_blockscale/gemm_a8w8_blockscale_tune.cu`**
```
using BlockwiseKernel = std::function<
torch::Tensor(torch::Tensor &, torch::Tensor &,
torch::Tensor &, torch::Tensor &,
torch::Tensor &)>;
```
