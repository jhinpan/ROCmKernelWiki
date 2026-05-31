# Diff summary

- **files changed:** 21
- **lines:** +1291 / -17
- **kernel-ish files:** 20

## Files (by churn)

- `csrc/kernels/mhc_kernels.cu`  (+566/-0)
- `op_tests/test_mhc.py`  (+512/-0)
- `aiter/ops/mhc.py`  (+118/-0)
- `csrc/include/mhc.h`  (+27/-0)
- `csrc/include/rocm_ops.hpp`  (+25/-0)
- `aiter/jit/optCompilerConfig.json`  (+15/-0)
- `csrc/pybind/mhc_pybind.cu`  (+10/-0)
- `aiter/dist/parallel_state.py`  (+2/-2)
- `aiter/ops/causal_conv1d.py`  (+2/-2)
- `csrc/kernels/causal_conv1d_update.cu`  (+2/-2)
- `op_tests/test_moe_sorting.py`  (+2/-2)
- `aiter/dist/communication_op.py`  (+1/-1)
- `csrc/include/causal_conv1d.h`  (+1/-1)
- `csrc/py_itfs_cu/asm_a8w8_blockscale_bpreshuffle.cu`  (+1/-1)
- `csrc/pybind/causal_conv1d_update_pybind.cu`  (+1/-1)

## Key added lines (kernel files)

**`aiter/__init__.py`**
```
from .ops.mhc import *  # noqa: F403,E402
```

**`aiter/ops/causal_conv1d.py`**
```
- Padding token handling: conv_state_indices[i] == pad_slot_id -> skip processing
```

**`aiter/ops/mhc.py`**
```
import torch
import math
from torch import Tensor
from aiter import dtypes
```

**`csrc/include/mhc.h`**
```
namespace aiter {
void mhc_pre_gemm_sqrsum(torch::Tensor& out,    // (split_k, m, hc_mult3) / (m, hc_mult3)
torch::Tensor& sqrsum, // (split_k, m) / (m)
torch::Tensor& x,      // (m, hc_hidden_size)
```

**`csrc/include/rocm_ops.hpp`**
```
m.def("mhc_pre_gemm_sqrsum",                \
&aiter::mhc_pre_gemm_sqrsum,          \
"mhc_pre_gemm_sqrsum",                \
py::arg("out"),                       \
```
