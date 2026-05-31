# Diff summary

- **files changed:** 11
- **lines:** +1347 / -258
- **kernel-ish files:** 10

## Files (by churn)

- `hsa/gfx942/fmoe_2stages/tune.py`  (+1201/-181)
- `csrc/py_itfs_cu/asm_fmoe.cu`  (+50/-39)
- `aiter/fused_moe_bf16_asm.py`  (+36/-19)
- `aiter/fused_moe.py`  (+21/-2)
- `aiter/utility/mp_tuner.py`  (+10/-8)
- `aiter/configs/tuned_fmoe.csv`  (+7/-7)
- `csrc/ck_gemm_moe_2stages_codegen/gemm_moe_ck2stages_common.py`  (+11/-2)
- `aiter/ops/moe_op.py`  (+4/-0)
- `csrc/include/moe_op.h`  (+3/-0)
- `csrc/include/rocm_ops.hpp`  (+3/-0)
- `op_tests/test_moe_blockscale.py`  (+1/-0)

## Key added lines (kernel files)

**`aiter/fused_moe.py`**
```
aiter.fmoe_int8_g1u0(
sorted_ids,
sorted_weights,
sorted_expert_ids,
```

**`aiter/fused_moe_bf16_asm.py`**
```
fmoe_func = aiter.fmoe_int8_g1u0(
sorted_ids,
sorted_weights,
sorted_expert_ids,
```

**`aiter/ops/moe_op.py`**
```
kernelName: str,
kernelName: str,
kernelName: str,
torch.int4: "i4",
```

**`aiter/utility/mp_tuner.py`**
```
if ref is None and not fast_mode or (ref_func is not None and fast_mode):
```

**`csrc/ck_gemm_moe_2stages_codegen/gemm_moe_ck2stages_common.py`**
```
Adtype in bit8_list
and Bdtype in bit4_list
and (Adtype == "F8" or Adtype == "f8")
Adtype in bit8_list
```
