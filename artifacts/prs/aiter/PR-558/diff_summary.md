# Diff summary

- **files changed:** 26
- **lines:** +514 / -433
- **kernel-ish files:** 18

## Files (by churn)

- `aiter/ops/triton/gemm_a8wfp4.py`  (+75/-128)
- `aiter/ops/triton/batched_gemm_afp4wfp4.py`  (+78/-98)
- `aiter/ops/triton/configs/gemm/MI350X-BATCHED_GEMM-AFP4WFP4.json`  (+82/-0)
- `aiter/ops/triton/gemm_a8w8_blockscale.py`  (+38/-35)
- `aiter/ops/triton/batched_gemm_bf16.py`  (+32/-29)
- `aiter/ops/triton/batched_gemm_a8w8.py`  (+32/-28)
- `aiter/ops/triton/configs/gemm/MI350X-GEMM-A8WFP4.json`  (+16/-15)
- `op_tests/triton_tests/test_gemm_a8w8.py`  (+6/-19)
- `aiter/ops/triton/configs/gemm/MI300X-BATCHED_GEMM-A16W16.json`  (+24/-0)
- `aiter/ops/triton/configs/gemm/MI300X-BATCHED_GEMM-A8W8.json`  (+24/-0)
- `aiter/ops/triton/configs/gemm/MI350X-BATCHED_GEMM-A16W16.json`  (+24/-0)
- `aiter/ops/triton/configs/gemm/MI350X-BATCHED_GEMM-A8W8.json`  (+24/-0)
- `op_tests/triton_tests/test_batched_gemm_a8w8.py`  (+4/-17)
- `op_tests/triton_tests/test_batched_gemm_bf16.py`  (+4/-17)
- `op_tests/triton_tests/test_gemm_a8w8_blockscale.py`  (+4/-17)

## Key added lines (kernel files)

**`aiter/ops/triton/batched_gemm_a8w8.py`**
```
from typing import Optional
import functools
import json
import aiter.ops.triton.utils.arch_info as arch_info
```

**`aiter/ops/triton/batched_gemm_afp4wfp4.py`**
```
import functools
import json
import aiter.ops.triton.utils.arch_info as arch_info
from aiter.ops.triton.utils.core import AITER_TRITON_CONFIGS_PATH
```

**`aiter/ops/triton/batched_gemm_bf16.py`**
```
from typing import Optional
import functools
import json
import aiter.ops.triton.utils.arch_info as arch_info
```

**`aiter/ops/triton/gemm_a8w8_blockscale.py`**
```
from typing import Optional
import functools
import json
import aiter.ops.triton.utils.arch_info as arch_info
```

**`aiter/ops/triton/gemm_a8wfp4.py`**
```
import functools
import json
import aiter.ops.triton.utils.arch_info as arch_info
from aiter.ops.triton.utils.core import AITER_TRITON_CONFIGS_PATH
```
