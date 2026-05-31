# Diff summary

- **files changed:** 8
- **lines:** +5000 / -12
- **kernel-ish files:** 6

## Files (by churn)

- `kernels/a8w4_moe_gemm_2stage.py`  (+3846/-0)
- `tests/kernels/test_a8w4_moe_gemm_2stage.py`  (+695/-0)
- `bench_moe_prefill.sh`  (+132/-0)
- `bench_moe.sh`  (+125/-0)
- `kernels/moe_gemm_2stage.py`  (+104/-11)
- `kernels/mfma_preshuffle_pipeline.py`  (+67/-0)
- `tests/utils.py`  (+22/-1)
- `python/flydsl/expr/rocdl/__init__.py`  (+9/-0)

## Key added lines (kernel files)

**`kernels/a8w4_moe_gemm_2stage.py`**
```
"""MoE GEMM stage1/stage2 kernel implementations — **a8w4smooth-only**.
This is a dedicated, API-restricted variant of `kernels.moe_gemm_2stage`.
The compiler infrastructure is shared with the multi-dtype kernel, but the
public `compile_*` entry points refuse any `in_dtype` other than
```

**`kernels/mfma_preshuffle_pipeline.py`**
```
@dataclass(frozen=True)
class PreshuffleBLayout:
"""Container returned by `make_preshuffle_b_layout`."""
layout_b: object
```

**`kernels/moe_gemm_2stage.py`**
```
mfma_i32_k64 = None
if is_int8 and _is_gfx950:
mfma_i32_k64 = getattr(rocdl, "mfma_i32_16x16x64_i8", None)
_use_int8_k64 = is_int8 and _is_gfx950 and (mfma_i32_k64 is not None)
```

**`python/flydsl/expr/rocdl/__init__.py`**
```
_ods_mfma_i32_16x16x64_i8 = globals().get("mfma_i32_16x16x64_i8", None)
@traced_op
def mfma_i32_16x16x64_i8(result_type, operands, *, loc=None, ip=None):
if _ods_mfma_i32_16x16x64_i8 is None:
```

**`tests/kernels/test_a8w4_moe_gemm_2stage.py`**
```
"""Self-contained a8w4smooth MoE GEMM 2-stage tests.
Exercises `kernels.a8w4_moe_gemm_2stage` (the a8w4smooth-only kernel module)
via standalone runner functions defined here.  All a8w4smooth-specific helpers
(weight generation, packing, dequant) live in this file so that the legacy
```
