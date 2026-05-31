# Diff summary

- **files changed:** 14
- **lines:** +18 / -10
- **kernel-ish files:** 2

## Files (by churn)

- `csrc/py_itfs_cu/asm_fmoe.cu`  (+9/-9)
- `hsa/gfx942/fmoe/gelu/fmoe_bf16_blockscaleFp8_g1u1_gelu.csv`  (+4/-0)
- `hsa/gfx942/fmoe/silu/fmoe_bf16_blockscaleFp8_g1u1_silu.csv`  (+4/-0)
- `aiter/fused_moe.py`  (+1/-1)
- `hsa/gfx942/fmoe/gelu/fmoe_bf16_blockscaleFp8_g1u1_novs_gelu_1tg_32x128.co`  (+0/-0)
- `hsa/gfx942/fmoe/gelu/fmoe_bf16_blockscaleFp8_g1u1_novs_gelu_1tg_ps_32x128.co`  (+0/-0)
- `hsa/gfx942/fmoe/gelu/fmoe_bf16_blockscaleFp8_g1u1_vs_gelu_1tg_32x128.co`  (+0/-0)
- `hsa/gfx942/fmoe/gelu/fmoe_bf16_blockscaleFp8_g1u1_vs_gelu_1tg_ps_32x128.co`  (+0/-0)
- `hsa/gfx942/fmoe/gelu/fmoe_bf16_pertokenFp8_g1u1_vs_gelu_1tg_32x128.co`  (+0/-0)
- `hsa/gfx942/fmoe/silu/fmoe_bf16_blockscaleFp8_g1u1_novs_silu_1tg_32x128.co`  (+0/-0)
- `hsa/gfx942/fmoe/silu/fmoe_bf16_blockscaleFp8_g1u1_novs_silu_1tg_ps_32x128.co`  (+0/-0)
- `hsa/gfx942/fmoe/silu/fmoe_bf16_blockscaleFp8_g1u1_vs_silu_1tg_32x128.co`  (+0/-0)
- `hsa/gfx942/fmoe/silu/fmoe_bf16_blockscaleFp8_g1u1_vs_silu_1tg_ps_32x128.co`  (+0/-0)
- `hsa/gfx942/fmoe/silu/fmoe_bf16_pertokenFp8_g1u1_vs_silu_1tg_32x128.co`  (+0/-0)

## Key added lines (kernel files)

**`aiter/fused_moe.py`**
```
run_1stage = True and (inter_dim % 128 == 0)
```

**`csrc/py_itfs_cu/asm_fmoe.cu`**
```
std::string arch_id         = get_gpu_arch();
std::string selectedKl      = kernel_name.empty() ? "" : arch_id + kernel_name;
if(el.first.find(arch_id) != 0)
continue;
```
