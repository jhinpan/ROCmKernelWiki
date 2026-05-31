# Diff summary

- **files changed:** 11 (diff was byte-capped; summary is partial)
- **lines:** +1439 / -637
- **kernel-ish files:** 5

## Files (by churn)

- `aiter/ops/flydsl/kernels/mixed_moe_gemm_2stage.py`  (+965/-487)
- `aiter/configs/model_configs/kimik2_fp4_tuned_fmoe.csv`  (+128/-128)
- `aiter/ops/flydsl/kernels/mfma_epilogues.py`  (+179/-1)
- `aiter/configs/model_configs/kimik2_fp8fp4_tuned_fmoe.csv`  (+65/-0)
- `aiter/fused_moe.py`  (+36/-15)
- `aiter/configs/model_configs/kimik2_fp8fp4_untuned_fmoe.csv`  (+33/-0)
- `aiter/configs/model_configs/gptoss_fp8fp4_tuned_fmoe.csv`  (+15/-0)
- `aiter/configs/model_configs/dsv3_fp4_tuned_fmoe.csv`  (+6/-6)
- `aiter/configs/model_configs/gptoss_fp8fp4_untuned_fmoe.csv`  (+8/-0)
- `aiter/jit/utils/moe_recipes.py`  (+3/-0)
- `aiter/ops/flydsl/kernels/mfma_preshuffle_pipeline.py`  (+1/-0)

## Key added lines (kernel files)

**`aiter/fused_moe.py`**
```
bf16_fp8_bound = 256
fuse_quant: str = ""
out_scale=None,
out_scale_sorted=None,
```

**`aiter/jit/utils/moe_recipes.py`**
```
if activation == "swiglu":
```

**`aiter/ops/flydsl/kernels/mfma_epilogues.py`**
```
from flydsl._mlir.dialects.arith import CmpIPredicate
lds_out_split=None,
lds_row_offset=None,
if lds_out_split is not None:
```

**`aiter/ops/flydsl/kernels/mfma_preshuffle_pipeline.py`**
```
"PreshuffleScaleLayout",
```

**`aiter/ops/flydsl/kernels/mixed_moe_gemm_2stage.py`**
```
from enum import Enum
class GateMode(str, Enum):
"""Gate/Up computation strategy for stage1 GEMM.
SEPARATED:      Two separate B-tile streams (gate + up), default mode.
```
