# Diff summary

- **files changed:** 7
- **lines:** +4504 / -20
- **kernel-ish files:** 7

## Files (by churn)

- `kernels/mixed_moe_gemm_2stage.py`  (+2378/-0)
- `kernels/mixed_preshuffle_gemm.py`  (+1089/-0)
- `tests/kernels/utils/fp4_utils.py`  (+687/-0)
- `tests/kernels/test_preshuffle_gemm.py`  (+216/-18)
- `flydsl/src/flydsl/lang/ir/types.py`  (+61/-0)
- `kernels/mfma_preshuffle_pipeline.py`  (+53/-1)
- `flydsl/src/flydsl/dialects/ext/arith.py`  (+20/-1)

## Key added lines (kernel files)

**`flydsl/src/flydsl/dialects/ext/arith.py`**
```
def index_cast_ui(target_type: Type, value: Union["ArithValue", Value, int], *, loc: Location = None) -> "ArithValue":
"""Cast between index and unsigned integer types.
target_type: Target type (index or unsigned integer type)
value: Value to cast
```

**`flydsl/src/flydsl/lang/ir/types.py`**
```
@property
def ui8(self) -> ir.Type:
return ir.IntegerType.get_unsigned(8)
@property
```

**`kernels/mfma_preshuffle_pipeline.py`**
```
def make_preshuffle_scale_layout(
c_mn: ir.Value,
c_k: ir.Value,
mn_pack: int = 2,
```

**`kernels/mixed_moe_gemm_2stage.py`**
```
"""MoE GEMM stage1/stage2 kernel implementations (FLIR MFMA FP8/FP16).
This module intentionally contains the **kernel builder code** for:
- `moe_gemm1` (stage1)
- `moe_gemm2` (stage2)
```

**`kernels/mixed_preshuffle_gemm.py`**
```
"""Preshuffle GEMM kernel implementations (FLIR MFMA FP8/INT8).
This module intentionally contains the **kernel builder code** for the preshuffle GEMM,
extracted from `tests/kernels/test_preshuffle_gemm.py` in the same style as
`kernels/moe_gemm_2stage.py`:
```
