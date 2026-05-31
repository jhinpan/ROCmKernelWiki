# Diff summary

- **files changed:** 26
- **lines:** +238 / -83
- **kernel-ish files:** 18

## Files (by churn)

- `aiter/jit/core.py`  (+96/-6)
- `aiter/ops/gemm_op_a4w4.py`  (+17/-13)
- `aiter/ops/gemm_op_a8w8.py`  (+15/-11)
- `aiter/ops/batched_gemm_op_a8w8.py`  (+19/-3)
- `aiter/ops/batched_gemm_op_bf16.py`  (+17/-3)
- `aiter/fused_moe.py`  (+15/-4)
- `csrc/ck_gemm_a4w4_blockscale/gemm_a4w4_blockscale_tune.py`  (+9/-7)
- `aiter/jit/optCompilerConfig.json`  (+7/-7)
- `csrc/ck_gemm_a8w8_bpreshuffle/gemm_a8w8_bpreshuffle_tune.py`  (+2/-10)
- `hsa/gfx950/fmoe_2stages/tune.py`  (+7/-2)
- `csrc/ck_batched_gemm_a8w8/gen_instances.py`  (+7/-0)
- `csrc/ck_batched_gemm_bf16/gen_instances.py`  (+6/-0)
- `csrc/ck_batched_gemm_a8w8/batched_gemm_a8w8_tune.py`  (+2/-2)
- `csrc/ck_batched_gemm_bf16/batched_gemm_bf16_tune.py`  (+2/-2)
- `csrc/ck_gemm_a8w8/gemm_a8w8_tune.py`  (+2/-2)

## Key added lines (kernel files)

**`aiter/fused_moe.py`**
```
import sys
from aiter.jit.core import (
AITER_CONFIG_FMOE_FILE,
get_asm_dir,
```

**`aiter/jit/core.py`**
```
import typing
from typing import Any, Callable, List, Optional, Union, get_args, get_origin
MainFunc: Callable,
FinalFunc: Optional[Callable] = None,
```

**`aiter/ops/batched_gemm_op_a8w8.py`**
```
AITER_CONFIG_A8W8_BATCHED_GEMM_FILE,
AITER_LOG_TUNED_CONFIG,
from aiter import logger
print("Loading CKBatchedGEMM config from:", AITER_CONFIG_A8W8_BATCHED_GEMM_FILE)
```

**`aiter/ops/batched_gemm_op_bf16.py`**
```
AITER_CONFIG_BF16_BATCHED_GEMM_FILE,
AITER_LOG_TUNED_CONFIG,
from aiter import logger
AITER_CONFIG_BF16_BATCHED_GEMM_FILE
```

**`aiter/ops/gemm_op_a4w4.py`**
```
import functools
import os
from typing import Optional
import pandas as pd
```
