# Diff summary

- **files changed:** 17
- **lines:** +120 / -61
- **kernel-ish files:** 9

## Files (by churn)

- `op_tests/test_moe.py`  (+82/-38)
- `csrc/py_itfs_cu/asm_fmoe.cpp`  (+16/-7)
- `aiter/ops/quant.py`  (+6/-6)
- `csrc/kernels/custom_kernels.cu`  (+4/-4)
- `csrc/py_itfs_ck/smoothquant_kernels.cu`  (+7/-1)
- `3rdparty/composable_kernel`  (+1/-1)
- `aiter/fused_moe.py`  (+1/-1)
- `aiter/fused_moe_bf16_asm.py`  (+1/-1)
- `aiter/fused_moe_gelu.py`  (+1/-1)
- `aiter/fused_moe_int8_a8w8.py`  (+1/-1)
- `hsa/fmoe_fp8_g1u1_multix_subGU_128.co`  (+0/-0)
- `hsa/fmoe_fp8_g1u1_multix_subGU_192.co`  (+0/-0)
- `hsa/fmoe_fp8_g1u1_multix_subGU_256.co`  (+0/-0)
- `hsa/fmoe_fp8_g1u1_multix_subGU_320.co`  (+0/-0)
- `hsa/fmoe_fp8_g1u1_multix_subGU_384.co`  (+0/-0)

## Key added lines (kernel files)

**`aiter/fused_moe.py`**
```
device_name = 'AMD_Instinct_OAM'  # TODO: need to update
```

**`aiter/fused_moe_bf16_asm.py`**
```
dtype=w1.dtype, device=device)
```

**`aiter/fused_moe_gelu.py`**
```
device_name = 'AMD_Instinct_OAM'  # TODO: need to update
```

**`aiter/fused_moe_int8_a8w8.py`**
```
device_name = 'AMD_Instinct_OAM'  # TODO: need to update
```

**`aiter/ops/quant.py`**
```
def pertoken_quant(x, y_scale_dtype=torch.float, x_scale=None, quant_dtype=torch.int8):
hidden_states = x
hidden_states = x.to(x_scale) * x_scale
y = (hidden_states / per_token_scale).to(dtype=quant_dtype)
```
