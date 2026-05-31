# Diff summary

- **files changed:** 13
- **lines:** +635 / -110
- **kernel-ish files:** 5

## Files (by churn)

- `aiter/ops/triton/_triton_kernels/gemm/basic/gemm_a8w8.py`  (+160/-89)
- `op_tests/triton_tests/gemm/basic/test_gemm_a8w8.py`  (+135/-0)
- `aiter/ops/triton/gemm/basic/gemm_a8w8.py`  (+62/-18)
- `aiter/ops/triton/utils/_triton/tunning/ut_a8w8_gemm.py`  (+45/-0)
- `aiter/ops/triton/configs/gemm/gfx942-GEMM-A8W8-N=128-K=2048.json`  (+38/-0)
- `aiter/ops/triton/configs/gemm/gfx942-GEMM-A8W8-N=128-K=4096.json`  (+38/-0)
- `aiter/ops/triton/configs/gemm/gfx942-GEMM-A8W8-N=128-K=6144.json`  (+38/-0)
- `aiter/ops/triton/configs/gemm/gfx942-GEMM-A8W8-N=256-K=2048.json`  (+38/-0)
- `aiter/ops/triton/configs/gemm/gfx942-GEMM-A8W8-N=256-K=4096.json`  (+38/-0)
- `aiter/ops/triton/configs/gemm/gfx942-GEMM-A8W8-N=256-K=6144.json`  (+38/-0)
- `aiter/ops/triton/configs/gemm/gfx942-GEMM-A8W8.json`  (+2/-1)
- `aiter/ops/triton/configs/gemm/gfx950-GEMM-A8W8.json`  (+2/-1)
- `aiter/ops/triton/utils/gemm_config_utils.py`  (+1/-1)

## Key added lines (kernel files)

**`aiter/ops/triton/_triton_kernels/gemm/basic/gemm_a8w8.py`**
```
from aiter.ops.triton.utils._triton.pid_preprocessing import pid_grid, remap_xcd
from aiter.ops.triton.utils.gemm_config_utils import (
get_gemm_config,
compute_splitk_params,
```

**`aiter/ops/triton/gemm/basic/gemm_a8w8.py`**
```
_gemm_a8w8_reduce_kernel,
skip_reduce: Optional[bool] = False,
x (torch.Tensor): Input matrix with shape (M, K).
w (torch.Tensor): Weight matrix with shape (N, K), internally transposed.
```

**`aiter/ops/triton/utils/_triton/tunning/ut_a8w8_gemm.py`**
```
import sys
from _utils import (
run_profile,
get_input_shape_and_config_list,
```

**`aiter/ops/triton/utils/gemm_config_utils.py`**
```
config["BLOCK_SIZE_K"] = config["BLOCK_SIZE_K"] // 2
```

**`op_tests/triton_tests/gemm/basic/test_gemm_a8w8.py`**
```
from aiter.ops.triton._triton_kernels.gemm.basic.gemm_a8w8 import _get_config
from aiter.ops.triton.utils.gemm_config_utils import compute_splitk_params
def get_splitk_x_vals():
(1, 1280, 8192),
```
