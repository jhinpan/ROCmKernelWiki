# Diff summary

- **files changed:** 8
- **lines:** +146 / -195
- **kernel-ish files:** 7

## Files (by churn)

- `aiter/fused_moe.py`  (+53/-131)
- `op_tests/test_moe_2stage.py`  (+68/-54)
- `csrc/py_itfs_cu/asm_fmoe.cpp`  (+13/-5)
- `aiter/ops/quant.py`  (+8/-2)
- `aiter/jit/core.py`  (+2/-1)
- `aiter/utility/fp4_utils.py`  (+1/-1)
- `csrc/py_itfs_cu/asm_moe_2stage.cpp`  (+1/-1)
- `hsa/gfx950/fmoe/silu/fmoe_fp4_g1u1_novs_subGU_256_test.co`  (+0/-0)

## Key added lines (kernel files)

**`aiter/fused_moe.py`**
```
BLOCK_SIZE_M = 32
q_dtype_a = dtypes.fp4x2 if quant_type == QuantType.per_1x32 else q_dtype_a
run_1stage = quant_type in [QuantType.per_128x128, QuantType.per_1x32]
if quant_type == QuantType.per_1x32:
```

**`aiter/jit/core.py`**
```
if get_gfx() == "gfx950" and int(os.getenv("AITER_FP4x2", "1")) > 0:
flags_hip += ["-D__Float4_e2m1fn_x2"]
```

**`aiter/ops/quant.py`**
```
shape_original = x.shape
x = x.view(-1, shape_original[-1])
y = y.view(*shape_original[:-1], -1)
(m + 255) // 256 * 256,
```

**`aiter/utility/fp4_utils.py`**
```
x = x.repeat_interleave(2, dim=-1)
```

**`csrc/py_itfs_cu/asm_fmoe.cpp`**
```
void set_4bit(bool is_4bit_)
is_int4 = is_4bit_;
int dim = w2.size(1);
int inter_dim = w2.size(2) * (w2.size(1) / w1.size(2));
```
