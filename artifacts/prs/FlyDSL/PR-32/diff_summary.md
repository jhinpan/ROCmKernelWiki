# Diff summary

- **files changed:** 13
- **lines:** +3243 / -1812
- **kernel-ish files:** 13

## Files (by churn)

- `tests/python/gpu/test_moe_gemm.py`  (+729/-858)
- `samples/moe_gemm_2stage.py`  (+1324/-0)
- `tests/python/gpu/test_preshuffle_gemm.py`  (+130/-669)
- `samples/preshuffle_gemm.py`  (+589/-0)
- `samples/mfma_preshuffle_pipeline.py`  (+306/-0)
- `tests/python/gpu/mfma_fp8_preshuffle_pipeline.py`  (+0/-196)
- `tests/python/gpu/test_ref.py`  (+81/-0)
- `samples/basic_layout.py`  (+0/-80)
- `pyflir/src/pyflir/dialects/ext/rocdl.py`  (+34/-0)
- `pyflir/src/pyflir/dialects/ext/arith.py`  (+21/-1)
- `pyflir/src/pyflir/dialects/ext/flir.py`  (+15/-6)
- `tests/utils.py`  (+12/-0)
- `tests/test_common.py`  (+2/-2)

## Key added lines (kernel files)

**`pyflir/src/pyflir/dialects/ext/arith.py`**
```
def sitofp(result_type: Type, value: Union["ArithValue", Value], *, loc: Location = None) -> "ArithValue":
"""Convert signed integer value to floating point.
result_type: Target floating point type (e.g., f32)
value: Signed integer value to convert
```

**`pyflir/src/pyflir/dialects/ext/flir.py`**
```
elem_ty = src_view.element_type
elem_ty_str = str(elem_ty)
is_i8 = False
is_i8 = IntegerType.isinstance(elem_ty) and (IntegerType(elem_ty).width == 8)
```

**`pyflir/src/pyflir/dialects/ext/rocdl.py`**
```
_ods_mfma_i32_16x16x32_i8 = mfma_i32_16x16x32_i8
_ods_raw_ptr_buffer_atomic_fadd = raw_ptr_buffer_atomic_fadd
def mfma_i32_16x16x32_i8_op(result_type, operands, *, loc=None, ip=None):
"""Return the op view (original behavior)."""
```

**`samples/mfma_preshuffle_pipeline.py`**
```
"""Shared MFMA preshuffle helpers (used by preshuffle GEMM + MoE kernels).
This module consolidates the common building blocks that were previously duplicated
- `samples/preshuffle_gemm.py`
- `samples/moe_gemm_2stage.py`
```

**`samples/moe_gemm_2stage.py`**
```
"""MoE GEMM stage1/stage2 kernel implementations (FLIR MFMA FP8).
This module intentionally contains the **kernel builder code** for:
- `moe_gemm1` (stage1)
- `moe_gemm2` (stage2)
```
