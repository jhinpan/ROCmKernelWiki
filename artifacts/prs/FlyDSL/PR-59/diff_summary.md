# Diff summary

- **files changed:** 16
- **lines:** +390 / -229
- **kernel-ish files:** 16

## Files (by churn)

- `flydsl/src/flydsl/dialects/ext/flir.py`  (+155/-21)
- `tests/kernels/test_eltwise_add.py`  (+32/-80)
- `tests/kernels/test_gpu_rocdsl.py`  (+47/-26)
- `tests/kernels/test_gpu_layout.py`  (+33/-26)
- `tests/kernels/test_gpu_simple.py`  (+29/-17)
- `tests/kernels/test_gpu_with_rocir_coords.py`  (+17/-7)
- `kernels/layernorm_kernel.py`  (+13/-9)
- `kernels/rmsnorm_kernel.py`  (+13/-9)
- `kernels/preshuffle_gemm.py`  (+14/-7)
- `tests/kernels/test_vec_add.py`  (+11/-9)
- `kernels/softmax_kernel.py`  (+11/-8)
- `tests/kernels/benchmark_common.py`  (+8/-6)
- `tests/kernels/test_preshuffle_gemm.py`  (+4/-1)
- `tests/kernels/test_layernorm.py`  (+1/-1)
- `tests/kernels/test_rmsnorm.py`  (+1/-1)

## Key added lines (kernel files)

**`flydsl/src/flydsl/dialects/ext/flir.py`**
```
if shape is None:
self.shape = ()
_shape = []
for s in shape:
```

**`kernels/layernorm_kernel.py`**
```
from _mlir import ir
DYN = ir.ShapedType.get_dynamic_size()
Input: lambda: T.memref(DYN, N, _state["elem_type"]),
Output: lambda: T.memref(DYN, N, _state["elem_type"]),
```

**`kernels/preshuffle_gemm.py`**
```
import os
a_rsrc = buffer_ops.create_buffer_resource(arg_a, max_size=False)
c_rsrc = buffer_ops.create_buffer_resource(arg_c, max_size=False)
scale_a_rsrc = buffer_ops.create_buffer_resource(arg_scale_a, max_size=False)
```

**`kernels/rmsnorm_kernel.py`**
```
from _mlir import ir
DYN = ir.ShapedType.get_dynamic_size()
Input: lambda: T.memref(DYN, N, _state["elem_type"]),
Output: lambda: T.memref(DYN, N, _state["elem_type"]),
```

**`kernels/softmax_kernel.py`**
```
DYN = ir.ShapedType.get_dynamic_size()
A: lambda: T.memref(DYN, N, _state["elem_type"]),
C: lambda: T.memref(DYN, N, _state["elem_type"]),
m_in: lambda: T.index(),
```
