# Diff summary

- **files changed:** 19
- **lines:** +1160 / -256
- **kernel-ish files:** 16

## Files (by churn)

- `csrc/kernels/quant_kernels.cu`  (+249/-76)
- `aiter/utility/fp4_utils.py`  (+204/-4)
- `csrc/include/ck_tile/vec_convert.h`  (+112/-77)
- `csrc/include/rocm_ops.hpp`  (+74/-66)
- `op_tests/test_gemm_a4w4.py`  (+128/-0)
- `csrc/py_itfs_cu/asm_gemm_a4w4.cpp`  (+119/-0)
- `op_tests/test_moe_sorting_mxfp4.py`  (+113/-0)
- `aiter/ops/quant.py`  (+39/-23)
- `aiter/ops/gemm_op_a4w4.py`  (+52/-0)
- `csrc/include/py_itfs_common.h`  (+20/-3)
- `csrc/include/asm_gemm_a4w4.h`  (+13/-0)
- `aiter/jit/optCompilerConfig.json`  (+12/-0)
- `csrc/include/dispatch_utils.h`  (+9/-2)
- `csrc/pybind/gemm_a4w4_asm_pybind.cu`  (+9/-0)
- `aiter/jit/core.py`  (+3/-4)

## Key added lines (kernel files)

**`aiter/__init__.py`**
```
from .ops.gemm_op_a4w4 import *
```

**`aiter/jit/core.py`**
```
if get_gfx() == "gfx950" and int(os.getenv("AITER_FP4x2", "1")) > 0:
flags_hip += ["-D__Float4_e2m1fn_x2"]
flags_hip = [el for el in flags_hip if hip_flag_checker(el)]
```

**`aiter/ops/gemm_op_a4w4.py`**
```
import torch
from torch import Tensor
from typing import Optional
import functools
```

**`aiter/ops/quant.py`**
```
F4E2M1_MAX = 6.0
dtypeMax = 2.0**MAX_POW2
scale_e8m0_biased = fp4_utils.f32_to_e8m0(max_abs / dtypeMax)
scale_f32 = fp4_utils.e8m0_to_f32(scale_e8m0_biased)
```

**`aiter/utility/fp4_utils.py`**
```
def f32_to_mxfp4(x):
def mxfp4_to_f32(x):
x = x.repeat_interleave(2, dim=1)
x[:, ::2] = x[:, ::2] & 0xF
```
