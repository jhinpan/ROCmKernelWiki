# Diff summary

- **files changed:** 7
- **lines:** +480 / -229
- **kernel-ish files:** 6

## Files (by churn)

- `kernels/fused_rope_cache_kernel.py`  (+124/-102)
- `.claude/skills/port-to-layout-api/SKILL.md`  (+188/-0)
- `kernels/layernorm_kernel.py`  (+52/-45)
- `kernels/rmsnorm_kernel.py`  (+45/-32)
- `kernels/softmax_kernel.py`  (+39/-28)
- `tests/kernels/test_quant.py`  (+26/-20)
- `tests/kernels/test_vec_add.py`  (+6/-2)

## Key added lines (kernel files)

**`kernels/fused_rope_cache_kernel.py`**
```
from flydsl.expr import vector, range_constexpr
elem_off = ArithValue(crd2idx(coord, layout)).index_cast(T.i32)
def _make_rope_copy_helpers(elem_type, elem_bits):
"""Build copy atom and register types for RoPE vector loads/stores."""
```

**`kernels/layernorm_kernel.py`**
```
from flydsl.expr.arith import ArithValue
wave_idx = ArithValue(wave).index_cast(T.index)
lane_safe_idx = ArithValue(lane_safe).index_cast(T.index)
Input_buf = fx.rocdl.make_buffer_tensor(Input)
```

**`kernels/rmsnorm_kernel.py`**
```
from flydsl.expr.arith import ArithValue
wave_idx = ArithValue(wave).index_cast(T.index)
lane_safe_idx = ArithValue(lane_safe).index_cast(T.index)
Input_buf = fx.rocdl.make_buffer_tensor(Input)
```

**`kernels/softmax_kernel.py`**
```
from flydsl.expr.arith import ArithValue
wave_idx = ArithValue(wave).index_cast(T.index)
lane_safe_idx = ArithValue(lane_safe).index_cast(T.index)
A_buf = fx.rocdl.make_buffer_tensor(A)
```

**`tests/kernels/test_quant.py`**
```
Input_buf = fx.rocdl.make_buffer_tensor(Input)
out_rsrc = buffer_ops.create_buffer_resource(Output, max_size=True)
row_in = fx.slice(Input_buf, (bid, None))
in_div = fx.logical_divide(row_in, fx.make_layout(VEC_WIDTH, 1))
```
