# Diff summary

- **files changed:** 6
- **lines:** +788 / -33
- **kernel-ish files:** 6

## Files (by churn)

- `kernels/fused_rope_cache_kernel.py`  (+408/-0)
- `tests/kernels/test_fused_rope_cache.py`  (+363/-0)
- `kernels/kernels_common.py`  (+14/-0)
- `kernels/layernorm_kernel.py`  (+1/-11)
- `kernels/rmsnorm_kernel.py`  (+1/-11)
- `kernels/softmax_kernel.py`  (+1/-11)

## Key added lines (kernel files)

**`kernels/fused_rope_cache_kernel.py`**
```
"""Fused RoPE + KV Cache kernel builder using the @flyc.kernel API.
Fuses 3 operations into two kernel launches:
Kernel 1 (Q RoPE):     Q → rotate → Q_out
Kernel 2 (K+V cache):  K → rotate → K_out + key_cache;  V → value_cache
```

**`kernels/kernels_common.py`**
```
def dtype_to_elem_type(dtype_str: str):
"""Map a dtype string to its MLIR scalar type.
Supported: 'f32', 'f16', 'bf16'.
if dtype_str == "f32":
```

**`kernels/layernorm_kernel.py`**
```
from kernels.kernels_common import dtype_to_elem_type, get_warp_size
```

**`kernels/rmsnorm_kernel.py`**
```
from kernels.kernels_common import dtype_to_elem_type, get_warp_size
```

**`kernels/softmax_kernel.py`**
```
from kernels.kernels_common import dtype_to_elem_type, get_warp_size
```
