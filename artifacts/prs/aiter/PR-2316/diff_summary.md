# Diff summary

- **files changed:** 6
- **lines:** +28 / -35
- **kernel-ish files:** 6

## Files (by churn)

- `op_tests/triton_tests/fusions/test_fused_kv_cache.py`  (+6/-13)
- `op_tests/triton_tests/attention/test_pa_decode.py`  (+12/-6)
- `op_tests/triton_tests/fusions/test_fused_bmm_rope_kv_cache.py`  (+4/-8)
- `op_tests/triton_tests/moe/test_moe.py`  (+2/-5)
- `aiter/ops/triton/_triton_kernels/flash_attn_triton_amd/utils.py`  (+2/-2)
- `op_tests/triton_tests/normalization/test_rmsnorm.py`  (+2/-1)

## Key added lines (kernel files)

**`aiter/ops/triton/_triton_kernels/flash_attn_triton_amd/utils.py`**
```
CDNA_ARCHS = frozenset({"gfx908", "gfx90a", "gfx940", "gfx941", "gfx942", "gfx950", "gfx1250"})
FP8_ARCHS = frozenset({"gfx942", "gfx950", "gfx1250"})
```

**`op_tests/triton_tests/attention/test_pa_decode.py`**
```
import aiter.ops.triton.utils._triton.arch_info as arch_info
import torch
import triton.language as tl
from aiter.utility.dtypes import fp8
```

**`op_tests/triton_tests/fusions/test_fused_bmm_rope_kv_cache.py`**
```
from aiter.utility.dtypes import fp8
cache_dtype_actual = fp8
from aiter.utility.dtypes import fp8
cache_dtype_actual = fp8
```

**`op_tests/triton_tests/fusions/test_fused_kv_cache.py`**
```
from aiter.utility.dtypes import fp8
cache_dtype_actual = fp8
from aiter.utility.dtypes import fp8
cache_dtype_actual = fp8
```

**`op_tests/triton_tests/moe/test_moe.py`**
```
from aiter.utility.dtypes import fp8
fp8_type = fp8
```
