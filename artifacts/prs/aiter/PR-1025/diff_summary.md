# Diff summary

- **files changed:** 7
- **lines:** +21 / -15
- **kernel-ish files:** 7

## Files (by churn)

- `csrc/py_itfs_cu/asm_fmoe.cu`  (+9/-7)
- `aiter/fused_moe.py`  (+2/-2)
- `aiter/utility/fp4_utils.py`  (+2/-2)
- `csrc/ck_gemm_moe_2stages_codegen/gemm_moe_ck2stages.cu`  (+2/-2)
- `op_tests/test_moe_2stage.py`  (+4/-0)
- `aiter/ops/moe_op.py`  (+1/-1)
- `aiter/ops/quant.py`  (+1/-1)

## Key added lines (kernel files)

**`aiter/fused_moe.py`**
```
elif q_type != QuantType.per_1x32:
dtypes.fp4x2,
```

**`aiter/ops/moe_op.py`**
```
dtypes.fp4x2: "fp4x2",
```

**`aiter/ops/quant.py`**
```
return y, scale
```

**`aiter/utility/fp4_utils.py`**
```
x = x.view(dtypes.fp4x2)  # to(fp32) for this datatype gives all 0 for torch...
```

**`csrc/ck_gemm_moe_2stages_codegen/gemm_moe_ck2stages.cu`**
```
if (hidden_states.dtype() == torch_fp4x2 && w1.dtype() == torch_fp4x2)
if (inter_states.dtype() == torch_fp4x2 && w2.dtype() == torch_fp4x2)
```
