# Diff summary

- **files changed:** 44
- **lines:** +806 / -851
- **kernel-ish files:** 43

## Files (by churn)

- `csrc/include/rocm_ops.hpp`  (+455/-0)
- `csrc/rocm_ops.cpp`  (+44/-242)
- `setup.py`  (+3/-111)
- `aiter/jit/optCompilerConfig.json`  (+46/-33)
- `aiter/ops/gradlib.py`  (+61/-0)
- `gradlib/include/hipbsolgemm.cuh`  (+57/-0)
- `gradlib/csrc/hipbsolgemm.cu`  (+16/-39)
- `gradlib/include/rocsolgemm.cuh`  (+46/-0)
- `csrc/pybind/moe_op_pybind.cu`  (+2/-42)
- `gradlib/csrc/rocsolgemm.cu`  (+4/-32)
- `csrc/pybind/norm_pybind.cu`  (+2/-32)
- `csrc/pybind/custom_all_reduce_pybind.cu`  (+2/-27)
- `csrc/pybind/moe_ck_2stages_pybind.cu`  (+2/-26)
- `csrc/pybind/cache_pybind.cu`  (+2/-25)
- `csrc/pybind/mha_varlen_bwd_pybind.cu`  (+2/-23)

## Key added lines (kernel files)

**`aiter/__init__.py`**
```
from .ops.gradlib import *
```

**`aiter/jit/core.py`**
```
AITER_GRADLIB_DIR = f'{AITER_ROOT_DIR}/gradlib'
rename_cpp_to_cu([f"{AITER_CSRC_DIR}/include"] + extra_include,
```

**`aiter/ops/gradlib.py`**
```
import os
import torch
from torch import Tensor
from typing import List, Optional
```

**`csrc/include/rocm_ops.hpp`**
```
m.def("silu_and_mul", &silu_and_mul, "Activation function used in SwiGLU."); \
m.def("gelu_and_mul", &gelu_and_mul, "Activation function used in GELU.");   \
m.def("gelu_tanh_and_mul", &gelu_tanh_and_mul, "Activation function used in GELU tanh.");
m.def("add", &aiter_add, "apply for add with transpose and broadcast.");    \
```

**`csrc/py_itfs_cu/fmha_bwd_pre_post_kernel.py`**
```
const char *AITER_ASM_DIR = std::getenv("AITER_ASM_DIR");
```
