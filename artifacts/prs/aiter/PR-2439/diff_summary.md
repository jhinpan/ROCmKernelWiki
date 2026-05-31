# Diff summary

- **files changed:** 24
- **lines:** +466 / -492
- **kernel-ish files:** 23

## Files (by churn)

- `aiter/jit/core.py`  (+169/-161)
- `csrc/py_itfs_cu/asm_fmoe.cu`  (+99/-99)
- `csrc/py_itfs_cu/asm_mla.cu`  (+41/-41)
- `csrc/py_itfs_cu/asm_flatmm_a8w8_blockscale.cu`  (+25/-32)
- `csrc/py_itfs_cu/asm_pa.cu`  (+25/-25)
- `csrc/py_itfs_cu/asm_layernorm.cu`  (+14/-14)
- `csrc/py_itfs_cu/asm_a8w8_blockscale_bpreshuffle.cu`  (+13/-13)
- `csrc/py_itfs_cu/asm_moe_2stage.cu`  (+12/-12)
- `aiter/ops/gemm_op_a8w8.py`  (+10/-11)
- `csrc/py_itfs_cu/asm_topksoftmax.cu`  (+6/-9)
- `aiter/utility/dtypes.py`  (+8/-6)
- `csrc/py_itfs_cu/asm_gemm_a4w4.cu`  (+7/-7)
- `csrc/py_itfs_cu/asm_gemm_a8w8.cu`  (+7/-7)
- `csrc/include/asm_flatmm_a8w8_blockscale.h`  (+0/-12)
- `csrc/py_itfs_cu/asm_mi350_a8w8_blockscale.cu`  (+6/-6)

## Key added lines (kernel files)

**`aiter/jit/core.py`**
```
Tensor                | POINTER(aiter_tensor_t) | aiter_tensor_t*
Optional[Tensor]      | POINTER(aiter_tensor_t) | aiter_tensor_t* (NULL if None)
from ..utility.dtypes import torch_to_aiter, aiter_tensor_t
argtypes.append(ctypes.POINTER(aiter_tensor_t))
```

**`aiter/ops/gemm_op_a8w8.py`**
```
@compile_ops(
"module_gemm_a8w8_blockscale_asm",
fc_name="flatmm_a8w8_blockscale_asm",
ffi_type="ctypes",
```

**`aiter/utility/aiter_types.py`**
```
class aiter_tensor_t(ctypes.Structure):
160  # must match sizeof(aiter_tensor_t) in csrc/include/aiter_tensor.h
assert ctypes.sizeof(aiter_tensor_t) == _EXPECTED_SIZEOF_AITER_TENSOR, (
f"aiter_tensor_t layout mismatch: Python sizeof={ctypes.sizeof(aiter_tensor_t)}, "
```

**`aiter/utility/dtypes.py`**
```
from .aiter_types import aiter_dtypes, aiter_tensor_t
def torch_to_aiter(tensor: torch.Tensor) -> aiter_tensor_t:
"""torch.Tensor -> aiter_tensor_t, zero-copy, points to the same GPU memory."""
assert tensor.is_cuda, "aiter_tensor_t only supports CUDA tensors"
```

**`csrc/include/aiter_tensor.h`**
```
struct aiter_tensor_t
```
