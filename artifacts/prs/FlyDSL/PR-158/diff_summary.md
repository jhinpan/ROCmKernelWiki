# Diff summary

- **files changed:** 9
- **lines:** +23 / -23
- **kernel-ish files:** 8

## Files (by churn)

- `docs/prebuilt_kernels_guide.md`  (+7/-7)
- `kernels/mixed_preshuffle_gemm.py`  (+3/-3)
- `kernels/moe_gemm_2stage.py`  (+3/-3)
- `kernels/preshuffle_gemm.py`  (+3/-3)
- `tests/kernels/test_moe_gemm.py`  (+2/-2)
- `tests/kernels/test_preshuffle_gemm.py`  (+2/-2)
- `tests/kernels/test_layernorm.py`  (+1/-1)
- `tests/kernels/test_rmsnorm.py`  (+1/-1)
- `tests/kernels/test_softmax.py`  (+1/-1)

## Key added lines (kernel files)

**`kernels/mixed_preshuffle_gemm.py`**
```
from flydsl.kernels.mfma_preshuffle_pipeline import (
from flydsl.kernels.mfma_epilogues import mfma_epilog
from flydsl.kernels.kernels_common import stream_ptr_to_async_token
```

**`kernels/moe_gemm_2stage.py`**
```
from flydsl.kernels.mfma_preshuffle_pipeline import (
from flydsl.kernels.mfma_epilogues import c_shuffle_epilog, default_epilog, mfma_epilog
from flydsl.kernels.kernels_common import stream_ptr_to_async_token
```

**`kernels/preshuffle_gemm.py`**
```
from flydsl.kernels.kernels_common import stream_ptr_to_async_token
from flydsl.kernels.mfma_preshuffle_pipeline import (
from flydsl.kernels.mfma_epilogues import mfma_epilog
```

**`tests/kernels/test_layernorm.py`**
```
from flydsl.kernels.layernorm_kernel import (
```

**`tests/kernels/test_moe_gemm.py`**
```
from flydsl.kernels.moe_gemm_2stage import (
from flydsl.kernels.moe_gemm_2stage import compile_moe_gemm1
```
