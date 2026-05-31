# Diff summary

- **files changed:** 8
- **lines:** +111 / -62
- **kernel-ish files:** 8

## Files (by churn)

- `aiter/ops/triton/gluon/gemm_a8w8.py`  (+36/-35)
- `aiter/ops/triton/gluon/triton_version.py`  (+30/-0)
- `aiter/ops/triton/gluon/pa_decode_gluon.py`  (+6/-19)
- `aiter/ops/triton/gluon/gemm_afp4wfp4.py`  (+14/-4)
- `op_tests/triton_tests/gemm/basic/test_gemm_a8w8.py`  (+14/-1)
- `op_tests/triton_tests/gemm/basic/test_gemm_afp4wfp4.py`  (+4/-2)
- `aiter/ops/triton/_triton_kernels/gemm/feed_forward/ff_a16w16_fused_ungated.py`  (+3/-1)
- `op_tests/triton_tests/attention/test_pa_decode.py`  (+4/-0)

## Key added lines (kernel files)

**`aiter/ops/triton/_triton_kernels/gemm/feed_forward/ff_a16w16_fused_ungated.py`**
```
y_mask = (offs_ym[:, None] < M) & (
(offs_k[None, :] + BLOCK_SIZE_K * k_cyclic_offset) < K
```

**`aiter/ops/triton/gluon/gemm_a8w8.py`**
```
import triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
```

**`aiter/ops/triton/gluon/gemm_afp4wfp4.py`**
```
import triton
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
```

**`aiter/ops/triton/gluon/pa_decode_gluon.py`**
```
import aiter
import aiter.ops.triton.utils._triton.arch_info as arch_info
import aiter.ops.triton.gluon.triton_version as tv
TRITON_VERSION_GE_3_6_0 = tl.constexpr(tv.TRITON_VERSION_GE_3_6_0)
```

**`aiter/ops/triton/gluon/triton_version.py`**
```
import triton
def parse_triton_version(version: str) -> tuple[int, ...]:
"""Parse version string into comparable tuple format, handling possible development version suffixes."""
version = version.split("+")[0].split("-")[0]
```
