# Diff summary

- **files changed:** 16 (diff was byte-capped; summary is partial)
- **lines:** +771 / -701
- **kernel-ish files:** 16

## Files (by churn)

- `kernels/moe_gemm_2stage.py`  (+186/-178)
- `kernels/moe_blockscale_2stage.py`  (+142/-120)
- `kernels/moe_gemm_2stage_mxscale_gfx1250.py`  (+106/-106)
- `kernels/blockscale_preshuffle_gemm.py`  (+75/-61)
- `kernels/gemm_fp8fp4_gfx1250.py`  (+64/-64)
- `kernels/flash_attn_func.py`  (+49/-49)
- `kernels/preshuffle_gemm.py`  (+39/-24)
- `kernels/mla_fwd_decode_m16x8_fp8_fp8.py`  (+20/-19)
- `kernels/layernorm_kernel.py`  (+20/-16)
- `kernels/custom_all_reduce_kernel.py`  (+16/-15)
- `kernels/hgemm_splitk.py`  (+15/-15)
- `kernels/pa_decode_fp8.py`  (+13/-13)
- `kernels/fused_rope_cache_kernel.py`  (+9/-9)
- `kernels/mixed_moe_gemm_2stage.py`  (+8/-8)
- `examples/04-preshuffle_gemm.py`  (+7/-2)

## Key added lines (kernel files)

**`examples/04-preshuffle_gemm.py`**
```
gA_k_stride = fx.get_scalar(gA_k.stride[2])
gB_k_stride = fx.get_scalar(gB_k.stride[2])
buffer_copy_128b,
soffset=next_k * gA_k_stride,
```

**`kernels/blockscale_preshuffle_gemm.py`**
```
from flydsl.expr import range_constexpr, const_expr
if const_expr(use_cshuffle_epilog):
def load_a(idx_i32, a_load_bytes_v):
if const_expr(a_load_bytes_v == 16):
```

**`kernels/custom_all_reduce_kernel.py`**
```
from flydsl.expr import const_expr
if const_expr(is_f32):
if const_expr(is_f32):
if const_expr(is_f32):
```

**`kernels/flash_attn_func.py`**
```
from flydsl.expr import arith, buffer_ops, const_expr, gpu, range_constexpr, rocdl, vector
if const_expr(dtype_str == "bf16"):
if const_expr(USE_K16):
if const_expr(USE_K16):
```

**`kernels/fused_rope_cache_kernel.py`**
```
from flydsl.expr import arith, vector, buffer_ops, range_constexpr, const_expr
if const_expr(VEC_WIDTH == 1):
if const_expr(pos_dtype == "i64"):
if const_expr(pos_dtype == "i64"):
```
