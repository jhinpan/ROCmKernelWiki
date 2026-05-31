# Diff summary

- **files changed:** 7
- **lines:** +936 / -202
- **kernel-ish files:** 6

## Files (by churn)

- `kernels/moe_gemm_2stage.py`  (+413/-166)
- `kernels/mfma_preshuffle_pipeline.py`  (+232/-0)
- `tests/kernels/test_moe_gemm.py`  (+155/-25)
- `tests/kernels/test_ref.py`  (+46/-9)
- `scripts/run_benchmark.sh`  (+54/-0)
- `tests/utils.py`  (+28/-1)
- `flydsl/src/flydsl/dialects/ext/arith.py`  (+8/-1)

## Key added lines (kernel files)

**`flydsl/src/flydsl/dialects/ext/arith.py`**
```
def bitcast(result_type: Type, value: Union["ArithValue", Value], *, loc: Location = None) -> "ArithValue":
"""Reinterpret-cast bits between types of the same width (e.g. f32 <-> i32)."""
loc = maybe_default_loc(loc)
val = _unwrap_value(value)
```

**`kernels/mfma_preshuffle_pipeline.py`**
```
def _i8x4_in_i32_to_bf16x4_i64(val_i32, arith, vector, scale_val=None):
"""Convert one i32 (4 signed int8 bytes) → 4 bf16 packed as i64.
Uses shift-based f32→bf16 truncation (``v_lshrrev_b32 dst, 16, src``)
instead of ``arith.truncf`` which on gfx942 expands to ~5 VALU per
```

**`kernels/moe_gemm_2stage.py`**
```
from flydsl.runtime.device import get_rocm_arch as get_hip_arch
load_b_raw_w4a16,
unpack_b_w4a16,
load_b_raw_w4a16_groupwise,
```

**`tests/kernels/test_moe_gemm.py`**
```
from tests.utils import pertoken_quant, shuffle_weight, shuffle_scale_for_int4
group_size: int = -1,
scale_w1_groups_in: Optional[torch.Tensor] = None,
if in_dtype not in ("fp8", "fp16", "bf16", "int8", "int8smooth", "int4", "int4_bf16"):
```

**`tests/kernels/test_ref.py`**
```
group_size: int = -1,
scale_w1_groups: torch.Tensor | None = None,
"""Return [tokens, topk, inter_dim] fp32.
group_size: -1 for per-row scale (uses scale_w1_flat), >0 for group-wise scale.
```
