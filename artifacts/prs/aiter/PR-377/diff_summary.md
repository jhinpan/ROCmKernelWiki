# Diff summary

- **files changed:** 14
- **lines:** +163 / -53
- **kernel-ish files:** 13

## Files (by churn)

- `op_tests/test_activation.py`  (+90/-0)
- `aiter/jit/core.py`  (+15/-11)
- `aiter/jit/utils/chip_info.py`  (+24/-0)
- `csrc/kernels/activation_kernels.cu`  (+6/-6)
- `aiter/utility/dtypes.py`  (+2/-8)
- `csrc/ck_gemm_a8w8_blockscale/gen_instances.py`  (+5/-4)
- `aiter/fused_moe.py`  (+3/-4)
- `aiter/ops/batched_gemm_op_a8w8.py`  (+3/-4)
- `aiter/ops/gemm_op_a8w8.py`  (+3/-4)
- `aiter/ops/batched_gemm_op_bf16.py`  (+3/-3)
- `csrc/py_itfs_ck/moe_kernels.cu`  (+3/-3)
- `csrc/py_itfs_ck/moe_sorting_kernels.cu`  (+3/-3)
- `aiter/mla.py`  (+2/-2)
- `3rdparty/composable_kernel`  (+1/-1)

## Key added lines (kernel files)

**`aiter/fused_moe.py`**
```
from aiter.jit.utils.chip_info import get_cu_num
cu_num = get_cu_num()
```

**`aiter/jit/core.py`**
```
sys.path.insert(0, f"{this_dir}/utils/")
from chip_info import get_gfx
CK_3RDPARTY_DIR = os.environ.get(
"CK_DIR", f"{AITER_ROOT_DIR}/3rdparty/composable_kernel"
```

**`aiter/jit/utils/chip_info.py`**
```
import os
import functools
@functools.lru_cache(maxsize=1)
def get_gfx():
```

**`aiter/mla.py`**
```
from .jit.utils.chip_info import get_cu_num
cu_num = get_cu_num()
```

**`aiter/ops/batched_gemm_op_a8w8.py`**
```
from ..jit.utils.chip_info import get_cu_num
cu_num = get_cu_num()
```
