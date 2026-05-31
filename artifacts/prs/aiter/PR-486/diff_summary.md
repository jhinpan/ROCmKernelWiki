# Diff summary

- **files changed:** 36
- **lines:** +2603 / -164
- **kernel-ish files:** 14

## Files (by churn)

- `aiter/ops/triton/gemm_afp4wfp4.py`  (+371/-87)
- `aiter/ops/triton/batched_gemm_afp4wfp4.py`  (+457/-0)
- `op_tests/triton_tests/test_batched_gemm_afp4wfp4.py`  (+141/-0)
- `op_tests/triton_tests/test_gemm_afp4wfp4.py`  (+96/-16)
- `aiter/ops/triton/configs/MI300X-GEMM-AFP4WFP4.json`  (+81/-0)
- `aiter/ops/triton/configs/MI350X-GEMM-AFP4WFP4-N=106496-K=16384.json`  (+75/-0)
- `aiter/ops/triton/configs/MI350X-GEMM-AFP4WFP4-N=13312-K=16384.json`  (+75/-0)
- `aiter/ops/triton/configs/MI350X-GEMM-AFP4WFP4-N=16384-K=13312.json`  (+75/-0)
- `aiter/ops/triton/configs/MI350X-GEMM-AFP4WFP4-N=16384-K=16384.json`  (+75/-0)
- `aiter/ops/triton/configs/MI350X-GEMM-AFP4WFP4-N=16384-K=2048.json`  (+75/-0)
- `aiter/ops/triton/configs/MI350X-GEMM-AFP4WFP4-N=16384-K=26624.json`  (+75/-0)
- `aiter/ops/triton/configs/MI350X-GEMM-AFP4WFP4-N=16384-K=4096.json`  (+75/-0)
- `aiter/ops/triton/configs/MI350X-GEMM-AFP4WFP4-N=16384-K=53248.json`  (+75/-0)
- `aiter/ops/triton/configs/MI350X-GEMM-AFP4WFP4-N=16384-K=6656.json`  (+75/-0)
- `aiter/ops/triton/configs/MI350X-GEMM-AFP4WFP4-N=16384-K=8192.json`  (+75/-0)

## Key added lines (kernel files)

**`aiter/ops/triton/batched_gemm_afp4wfp4.py`**
```
from typing import Optional
import os
import torch
import triton
```

**`aiter/ops/triton/gemm_a16w16.py`**
```
import functools
import json
import aiter.ops.triton.utils.arch_info as arch_info
from aiter.ops.triton.utils.core import AITER_TRITON_CONFIGS_PATH
```

**`aiter/ops/triton/gemm_a8w8.py`**
```
from typing import Optional
import functools
import json
import aiter.ops.triton.utils.arch_info as arch_info
```

**`aiter/ops/triton/gemm_afp4wfp4.py`**
```
import functools
import json
import aiter.ops.triton.utils.arch_info as arch_info
from aiter.ops.triton.utils.core import AITER_TRITON_CONFIGS_PATH
```

**`aiter/ops/triton/utils/arch_info.py`**
```
import triton
ARCH_TO_DEVICE = {
"gfx942": "MI300X",
"gfx950": "MI350X",
```
