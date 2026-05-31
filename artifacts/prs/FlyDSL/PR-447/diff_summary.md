# Diff summary

- **files changed:** 10
- **lines:** +308 / -436
- **kernel-ish files:** 10

## Files (by churn)

- `kernels/mla_fwd_decode_m16x8_fp8_fp8.py`  (+106/-206)
- `kernels/layernorm_kernel.py`  (+48/-63)
- `kernels/rmsnorm_kernel.py`  (+40/-52)
- `kernels/gemm_fp8fp4_gfx1250.py`  (+35/-42)
- `kernels/softmax_kernel.py`  (+29/-44)
- `kernels/fused_rope_cache_kernel.py`  (+17/-19)
- `kernels/kernels_common.py`  (+11/-6)
- `python/flydsl/expr/rocdl/__init__.py`  (+17/-0)
- `kernels/blockscale_preshuffle_gemm.py`  (+3/-3)
- `kernels/flash_attn_func.py`  (+2/-1)

## Key added lines (kernel files)

**`kernels/blockscale_preshuffle_gemm.py`**
```
combined = s_a_vecs[mi] * s_b_vecs[ni]
a0 = fx.Int64(-1)
a1 = fx.Int64(-1)
```

**`kernels/flash_attn_func.py`**
```
elem_dtype = dtype_to_elem_type(dtype_str)
elem_type = elem_dtype.ir_type
```

**`kernels/fused_rope_cache_kernel.py`**
```
from flydsl.expr import arith, buffer_ops, const_expr, range_constexpr, vector
elem_val = vec_val[0]
pos_elem_off = pid_t * 2
pair_lane = tid ^ fx.Int32(vecs_per_half)
```

**`kernels/gemm_fp8fp4_gfx1250.py`**
```
extract_lds_base_idx,
get_lds_memref,
pipeline_fence,
pipeline_fence_signal,
```

**`kernels/kernels_common.py`**
```
import flydsl.expr as fx
from flydsl._mlir.dialects import arith as _std_arith
from flydsl._mlir.dialects import builtin
from flydsl._mlir.dialects import gpu as _gpu
```
