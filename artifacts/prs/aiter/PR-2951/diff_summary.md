# Diff summary

- **files changed:** 13
- **lines:** +577 / -147
- **kernel-ish files:** 10

## Files (by churn)

- `aiter/fused_moe.py`  (+135/-38)
- `aiter/configs/model_configs/dsv4_fp8fp4_tuned_fmoe.csv`  (+166/-0)
- `aiter/ops/shuffle.py`  (+69/-35)
- `op_tests/test_moe_2stage.py`  (+50/-11)
- `aiter/configs/model_configs/dsv4_fp8fp4_untuned_fmoe.csv`  (+49/-0)
- `aiter/ops/flydsl/kernels/mixed_moe_gemm_2stage.py`  (+24/-22)
- `aiter/aot/flydsl/moe.py`  (+27/-12)
- `aiter/ops/flydsl/kernels/silu_and_mul_fq.py`  (+17/-8)
- `aiter/ops/flydsl/moe_common.py`  (+24/-0)
- `aiter/utility/fp4_utils.py`  (+2/-19)
- `aiter/ops/flydsl/moe_kernels.py`  (+11/-1)
- `aiter/configs/model_configs/gptoss_fp8fp4_tuned_fmoe.csv`  (+1/-1)
- `aiter/ops/flydsl/__init__.py`  (+2/-0)

## Key added lines (kernel files)

**`aiter/aot/flydsl/moe.py`**
```
q_dtype_w = row.get("q_dtype_w", "")
q_dtype_a = row.get("q_dtype_a", "")
q_type.strip().split(".")[-1] == "per_1x32"
and "float4_e2m1fn_x2" in q_dtype_w
```

**`aiter/fused_moe.py`**
```
from aiter.ops.flydsl.moe_common import GateMode
_ACT_TYPE_DISABLED_KEY = "__ignore__"
swiglu_limit=0.0,
gate_mode: Optional[str] = GateMode.SEPARATED.value,
```

**`aiter/ops/flydsl/__init__.py`**
```
from .moe_common import GateMode
"GateMode",
```

**`aiter/ops/flydsl/kernels/mixed_moe_gemm_2stage.py`**
```
from aiter.ops.flydsl.moe_common import (
GateMode,
)  # noqa: F401  re-exported for back-compat
swiglu_limit: float = 0.0,
```

**`aiter/ops/flydsl/kernels/silu_and_mul_fq.py`**
```
swiglu_limit: float = 0.0,
if const_expr(swiglu_limit != 0):
_limit = arith.constant(float(swiglu_limit), type=f32)
_neg_limit = arith.constant(-float(swiglu_limit), type=f32)
```
