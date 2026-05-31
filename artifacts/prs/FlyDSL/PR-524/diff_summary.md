# Diff summary

- **files changed:** 16 (diff was byte-capped; summary is partial)
- **lines:** +1398 / -1590
- **kernel-ish files:** 8

## Files (by churn)

- `kernels/moe_blockscale_2stage.py`  (+983/-1077)
- `kernels/moe_gemm_2stage.py`  (+289/-326)
- `kernels/preshuffle_gemm_v2.py`  (+51/-46)
- `kernels/fused_rope_cache_kernel.py`  (+29/-24)
- `kernels/rmsnorm_kernel.py`  (+14/-38)
- `kernels/layernorm_kernel.py`  (+8/-33)
- `.claude/skills/port-to-layout-api/SKILL.md`  (+3/-9)
- `.claude/skills/flydsl-kernel-authoring/SKILL.md`  (+4/-7)
- `.claude/skills/flydsl-tile-programming/SKILL.md`  (+2/-7)
- `README.md`  (+3/-5)
- `docs/quickstart.rst`  (+3/-5)
- `examples/01-vectorAdd.py`  (+3/-5)
- `kernels/silu_and_mul_fq.py`  (+3/-5)
- `.claude/skills/flydsl-internal-types-cleanup/SKILL.md`  (+1/-1)
- `docs/api/dsl.rst`  (+1/-1)

## Key added lines (kernel files)

**`examples/01-vectorAdd.py`**
```
rA = fx.make_rmem_tensor(fx.make_layout(1, 1), fx.Float32)
rB = fx.make_rmem_tensor(fx.make_layout(1, 1), fx.Float32)
rC = fx.make_rmem_tensor(fx.make_layout(1, 1), fx.Float32)
```

**`kernels/fused_rope_cache_kernel.py`**
```
from flydsl.expr import arith, buffer_ops, const_expr, range_constexpr
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec
raise ValueError(f"dtype_str must be 'bf16' or 'f16', got {dtype_str!r}")
```

**`kernels/layernorm_kernel.py`**
```
r = fx.make_rmem_tensor(VEC_WIDTH, elem_dtype)
r = fx.make_rmem_tensor(VEC_WIDTH, elem_dtype)
r = fx.make_rmem_tensor(1, elem_dtype)
r = fx.make_rmem_tensor(1, elem_dtype)
```

**`kernels/moe_blockscale_2stage.py`**
```
import functools
from flydsl._mlir.dialects import llvm, scf
from flydsl.compiler.kernel_function import CompilationContext
from flydsl.expr import arith, buffer_ops, const_expr, gpu, range_constexpr, rocdl, vector
```

**`kernels/moe_gemm_2stage.py`**
```
import functools
from flydsl.expr import arith, buffer_ops, const_expr, gpu, range_constexpr, rocdl, vector
supports_bf16_global_atomics,
from flydsl._mlir.dialects import llvm, scf
```
