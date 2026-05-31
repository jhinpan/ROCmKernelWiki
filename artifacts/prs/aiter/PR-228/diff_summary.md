# Diff summary

- **files changed:** 52 (diff was byte-capped; summary is partial)
- **lines:** +1986 / -3426
- **kernel-ish files:** 48

## Files (by churn)

- `aiter/fused_moe.py`  (+783/-890)
- `aiter/fused_moe_int8_a8w8.py`  (+0/-1183)
- `aiter/fused_moe_gelu.py`  (+0/-948)
- `aiter/jit/core.py`  (+219/-127)
- `aiter/test_common.py`  (+132/-70)
- `csrc/py_itfs_ck/moe_ck_2stages_gemm_impl/moe_ck_gemm.hpp`  (+140/-34)
- `aiter/ops/quant.py`  (+131/-25)
- `aiter/__init__.py`  (+32/-25)
- `csrc/include/moe_op.h`  (+33/-22)
- `aiter/ops/moe_op.py`  (+36/-15)
- `aiter/utility/mp_tuner.py`  (+45/-0)
- `csrc/include/rocm_ops.hpp`  (+31/-6)
- `aiter/ops/triton/quant.py`  (+15/-10)
- `aiter/jit/optCompilerConfig.json`  (+20/-4)
- `aiter/jit/cpp_extension.py`  (+18/-3)

## Key added lines (kernel files)

**`aiter/__init__.py`**
```
def getLogger():
global logger
if not logger.handlers:
logger.setLevel(logging.DEBUG)
```

**`aiter/fused_moe.py`**
```
import torch
import torch.nn.functional as F
import numpy as np
import time
```

**`aiter/fused_moe_bf16_asm.py`**
```
hidden_states, quant_dtype=w1.dtype)
hidden_states, quant_dtype=w1.dtype)
a1, a1_scale = aiter.per_tensor_quant_hip(a1, a1_scale, quant_dtype=w1.dtype)
a2, a2_scale = aiter.per_tensor_quant_hip(a2, a2_scale, quant_dtype=w2.dtype)
```

**`aiter/jit/core.py`**
```
import re
def mp_lock(
lockPath: str,
MainFunc: callable,
```

**`aiter/jit/cpp_extension.py`**
```
header_include_dirs=(
extra_include_paths
if extra_include_paths is not None
ignores=[
```
