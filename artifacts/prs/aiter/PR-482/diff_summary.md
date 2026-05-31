# Diff summary

- **files changed:** 9
- **lines:** +643 / -441
- **kernel-ish files:** 7

## Files (by churn)

- `csrc/py_itfs_cu/asm_fmoe.cu`  (+470/-325)
- `op_tests/test_moe_2stage.py`  (+67/-73)
- `aiter/fused_moe.py`  (+68/-15)
- `aiter/utility/fp4_utils.py`  (+28/-5)
- `aiter/ops/quant.py`  (+7/-21)
- `aiter/__init__.py`  (+2/-1)
- `csrc/py_itfs_cu/asm_moe_2stage.cu`  (+1/-1)
- `hsa/gfx950/fmoe/silu/fmoe_mxfp4_g1u1_vs_subGU_256.co`  (+0/-0)
- `hsa/gfx950/fmoe/silu/fmoe_mxfp4_g1u1_vs_subGU_512.co`  (+0/-0)

## Key added lines (kernel files)

**`aiter/__init__.py`**
```
from .jit import core as core
from .utility import dtypes as dtypes
```

**`aiter/fused_moe.py`**
```
from aiter.utility import fp4_utils
q_dtype_a = dtypes.fp4x2 if quant_type == QuantType.per_1x32 else q_dtype_a
run_1stage = quant_type in [QuantType.per_128x128, QuantType.per_1x32]
token_num = hidden_states.shape[0]
```

**`aiter/ops/quant.py`**
```
def per_1x32_f4_quant(x, scale=None, quant_dtype=dtypes.fp4x2, shuffle=False):
shape_original = x.shape
x = x.view(-1, shape_original[-1])
y = y.view(*shape_original[:-1], -1)
```

**`aiter/utility/fp4_utils.py`**
```
x = x.repeat_interleave(2, dim=-1)
x[..., ::2] = x[..., ::2] & 0xF
x[..., 1::2] = x[..., 1::2] >> 4
mxfp4_in_f32 = torch.tensor(mxfp4_list, dtype=torch.float32, device=x.device)
```

**`csrc/py_itfs_cu/asm_fmoe.cu`**
```
void* ptr_O;
void* ptr_X;
void* ptr_GU;
void* ptr_XC;
```
