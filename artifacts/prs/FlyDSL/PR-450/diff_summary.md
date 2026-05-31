# Diff summary

- **files changed:** 12
- **lines:** +464 / -851
- **kernel-ish files:** 12

## Files (by churn)

- `kernels/mla_fwd_decode_m16x8_fp8_fp8.py`  (+288/-702)
- `kernels/fused_rope_cache_kernel.py`  (+33/-39)
- `kernels/layernorm_kernel.py`  (+24/-36)
- `kernels/rmsnorm_kernel.py`  (+20/-29)
- `kernels/softmax_kernel.py`  (+17/-25)
- `python/flydsl/expr/rocdl.py`  (+27/-0)
- `python/flydsl/expr/utils/arith.py`  (+16/-4)
- `kernels/silu_and_mul_fq.py`  (+8/-10)
- `python/flydsl/utils/smem_allocator.py`  (+12/-2)
- `python/flydsl/expr/buffer_ops.py`  (+7/-3)
- `python/flydsl/expr/rocdl/__init__.py`  (+10/-0)
- `python/flydsl/compiler/ast_rewriter.py`  (+2/-1)

## Key added lines (kernel files)

**`kernels/fused_rope_cache_kernel.py`**
```
if tid < vecs_per_head:
is_first_half = tid < vecs_per_half
pair_lane = tid ^ vecs_per_half
pair_byte_addr = pair_lane * 4
```

**`kernels/layernorm_kernel.py`**
```
eps_c = EPS
off = WARP_SIZE // (2 << _sh_exp)
peer = w.shuffle_xor(off, WARP_SIZE)
if lane == 0:
```

**`kernels/mla_fwd_decode_m16x8_fp8_fp8.py`**
```
from flydsl.expr import math as fmath
from flydsl.expr.utils.arith import ArithValue
_gep = buffer_ops.get_element_ptr
addr_i64 = arith.index_cast(T.i64, byte_addr_index)
```

**`kernels/rmsnorm_kernel.py`**
```
eps_c = EPS
n_float = float(N)
off = WARP_SIZE // (2 << _sh_exp)
peer = w.shuffle_xor(off, WARP_SIZE)
```

**`kernels/silu_and_mul_fq.py`**
```
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith, buffer_ops, const_expr, range_constexpr, rocdl, vector
from flydsl.expr import math as fx_math
from flydsl.expr.arith import ArithValue
```
