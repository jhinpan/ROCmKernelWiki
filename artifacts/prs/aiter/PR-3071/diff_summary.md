# Diff summary

- **files changed:** 9
- **lines:** +1195 / -615
- **kernel-ish files:** 7

## Files (by churn)

- `aiter/ops/flydsl/kernels/preshuffle_gemm.py`  (+735/-235)
- `aiter/configs/a8w8_bpreshuffle_tuned_gemm.csv`  (+191/-191)
- `aiter/configs/model_configs/dsv3_a8w8_bpreshuffle_tuned_gemm.csv`  (+131/-131)
- `aiter/ops/flydsl/kernels/mfma_preshuffle_pipeline.py`  (+83/-31)
- `aiter/ops/flydsl/gemm_tune/flydsl_gemm_a8w8_bpreshuffle_common.py`  (+23/-17)
- `aiter/ops/gemm_op_a8w8.py`  (+8/-7)
- `aiter/aot/flydsl/gemm.py`  (+8/-3)
- `csrc/ck_gemm_a8w8_bpreshuffle/gemm_a8w8_bpreshuffle_tune.py`  (+9/-0)
- `aiter/ops/flydsl/gemm_kernels.py`  (+7/-0)

## Key added lines (kernel files)

**`aiter/aot/flydsl/gemm.py`**
```
r"(?P<lds_stage>\d+)x(?P<cshuffle>\d+)x(?P<async_copy>\d+)x"
r"(?P<waves_per_eu>\d+)x(?P<xcd_swizzle>\d+)_"
r"(?P<scheduler>[A-Za-z][A-Za-z0-9]*)$"
"xcd_swizzle": int(m.group("xcd_swizzle")),
```

**`aiter/ops/flydsl/gemm_kernels.py`**
```
xcd_swizzle: int = 0,
xcd_swizzle=int(xcd_swizzle),
_dummy_bias = torch.empty(0, dtype=Out.dtype, device=Out.device)
_dummy_bias,
```

**`aiter/ops/flydsl/gemm_tune/flydsl_gemm_a8w8_bpreshuffle_common.py`**
```
xcd_swizzle: int = 0  # 0=off, >0=group size for XCD remap
self.xcd_swizzle,
xcd_swizzle=0,
xcd_swizzle,
```

**`aiter/ops/flydsl/kernels/mfma_preshuffle_pipeline.py`**
```
from flydsl._mlir.dialects.arith import CmpIPredicate
layout_scale: object
stride_n0: object
stride_k0: object
```

**`aiter/ops/flydsl/kernels/preshuffle_gemm.py`**
```
from typing import Optional
from flydsl.expr import buffer_ops, const_expr, gpu, math, range_constexpr, rocdl
from .mfma_epilogues import mfma_epilog
_buffer_load_vec,
```
