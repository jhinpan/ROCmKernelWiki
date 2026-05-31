# Diff summary

- **files changed:** 75
- **lines:** +397 / -376
- **kernel-ish files:** 65

## Files (by churn)

- `aiter/jit/optCompilerConfig.json`  (+54/-54)
- `op_tests/test_pa.py`  (+39/-29)
- `op_tests/test_moe.py`  (+32/-21)
- `op_tests/test_communication.py`  (+17/-17)
- `gradlib/gradlib/GemmTuner.py`  (+15/-15)
- `op_tests/test_quant.py`  (+15/-15)
- `aiter/jit/core.py`  (+11/-11)
- `op_tests/test_kvcache.py`  (+11/-11)
- `aiter/fused_moe_bf16_asm.py`  (+10/-10)
- `op_tests/test_layernorm2dFusedAddQuant.py`  (+10/-10)
- `csrc/kernels/aiter_operator.cu`  (+9/-9)
- `op_tests/test_rmsnorm2dFusedAddQuant.py`  (+9/-9)
- `aiter/ops/communication.py`  (+7/-7)
- `csrc/ck_gemm_a8w8/README.md`  (+7/-7)
- `csrc/ck_gemm_a8w8/gemm_a8w8_tune.py`  (+7/-7)

## Key added lines (kernel files)

**`aiter/__init__.py`**
```
logger = logging.getLogger("aiter")
if importlib.util.find_spec('aiter_') is not None:
from aiter_ import *
from .ops.aiter_operator import *
```

**`aiter/dist/custom_all_reduce.py`**
```
import aiter as ops
from aiter import logger
```

**`aiter/dist/custom_all_reduce_utils.py`**
```
from aiter import logger
```

**`aiter/dist/parallel_state.py`**
```
from aiter import logger
```

**`aiter/dist/shm_broadcast.py`**
```
from aiter import logger
```
