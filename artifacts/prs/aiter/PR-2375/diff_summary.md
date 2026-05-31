# Diff summary

- **files changed:** 20
- **lines:** +2070 / -259
- **kernel-ish files:** 11

## Files (by churn)

- `aiter/utility/base_tuner.py`  (+856/-23)
- `gradlib/README.md`  (+172/-109)
- `csrc/ck_gemm_moe_2stages_codegen/gemm_moe_tune.py`  (+208/-0)
- `csrc/ck_gemm_a8w8_blockscale/gemm_a8w8_blockscale_tune.py`  (+76/-30)
- `csrc/ck_gemm_a8w8_bpreshuffle/gemm_a8w8_bpreshuffle_tune.py`  (+62/-20)
- `csrc/ck_gemm_a4w4_blockscale/gemm_a4w4_blockscale_tune.py`  (+66/-13)
- `gradlib/gradlib/GemmTuner.py`  (+77/-2)
- `aiter/utility/mp_tuner.py`  (+26/-47)
- `csrc/ck_gemm_a8w8/gemm_a8w8_tune.py`  (+53/-2)
- `csrc/ck_batched_gemm_a8w8/batched_gemm_a8w8_tune.py`  (+47/-7)
- `csrc/ck_batched_gemm_bf16/batched_gemm_bf16_tune.py`  (+45/-6)
- `csrc/ck_batched_gemm_a8w8/README.md`  (+47/-0)
- `csrc/ck_batched_gemm_bf16/README.md`  (+47/-0)
- `csrc/ck_gemm_a4w4_blockscale/README.md`  (+47/-0)
- `csrc/ck_gemm_a8w8/README.md`  (+47/-0)

## Key added lines (kernel files)

**`aiter/fused_moe.py`**
```
if not is_shuffled and not run_1stage:
logger.warning(
f"[fused_moe] tuned config found for {keys} but is_shuffled=False. "
"Tuned kernels are optimized for preshuffled weights (preshuffle_on). "
```

**`aiter/utility/base_tuner.py`**
```
import shutil
import tempfile
def _read_csv(filepath, **kwargs):
"""Read CSV with automatic cleanup of common formatting issues:
```

**`aiter/utility/mp_tuner.py`**
```
assert len(task_group) == len(
), f"shape_grouped: group count ({len(task_group)}) != in_datas count ({len(in_datas)})"
ref_data_index = list(range(len(task_group)))
import numpy as np
```

**`csrc/ck_batched_gemm_a8w8/batched_gemm_a8w8_tune.py`**
```
"config_env_name": "AITER_CONFIG_A8W8_BATCHED_GEMM",
def _clear_op_caches(self):
from aiter.ops.batched_gemm_op_a8w8 import get_CKBatchedGEMM_config
get_CKBatchedGEMM_config.cache_clear()
```

**`csrc/ck_batched_gemm_bf16/batched_gemm_bf16_tune.py`**
```
"config_env_name": "AITER_CONFIG_BF16_BATCHED_GEMM",
def _clear_op_caches(self):
from aiter.ops.batched_gemm_op_bf16 import get_CKBatchedGEMM_config
get_CKBatchedGEMM_config.cache_clear()
```
