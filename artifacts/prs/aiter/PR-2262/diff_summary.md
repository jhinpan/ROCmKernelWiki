# Diff summary

- **files changed:** 19
- **lines:** +388 / -43
- **kernel-ish files:** 5

## Files (by churn)

- `csrc/ck_gemm_moe_2stages_codegen/gemm_moe_tune.py`  (+100/-2)
- `aiter/configs/model_configs/qwen3next_80b_fp8_tuned_fmoe.csv`  (+97/-0)
- `aiter/configs/model_configs/qwen3next_80b_fp8_untuned_fmoe.csv`  (+97/-0)
- `aiter/fused_moe.py`  (+38/-21)
- `aiter/utility/base_tuner.py`  (+28/-7)
- `aiter/jit/core.py`  (+15/-11)
- `csrc/py_itfs_cu/asm_fmoe.cu`  (+3/-2)
- `hsa/gfx950/fmoe/gelu/fmoe_bf16_blockscaleBf16_g1u1_gelu.csv`  (+5/-0)
- `hsa/gfx950/fmoe/silu/fmoe_bf16_blockscaleBf16_g1u1_silu.csv`  (+5/-0)
- `hsa/gfx950/fmoe/gelu/fmoe_bf16_blockscaleBf16_g1u1_ps_gelu_32x128.co`  (+0/-0)
- `hsa/gfx950/fmoe/gelu/fmoe_bf16_blockscaleBf16_g1u1_ps_gelu_32x256.co`  (+0/-0)
- `hsa/gfx950/fmoe/gelu/fmoe_bf16_blockscaleBf16_g1u1_vs_1tg_gelu_16x128.co`  (+0/-0)
- `hsa/gfx950/fmoe/gelu/fmoe_bf16_blockscaleBf16_g1u1_vs_1tg_gelu_16x256.co`  (+0/-0)
- `hsa/gfx950/fmoe/silu/fmoe_bf16_blockscaleBf16_g1u1_ps_silu_32x128.co`  (+0/-0)
- `hsa/gfx950/fmoe/silu/fmoe_bf16_blockscaleBf16_g1u1_ps_silu_32x256.co`  (+0/-0)

## Key added lines (kernel files)

**`aiter/fused_moe.py`**
```
xbf16=False,
if xbf16:
a1_scale = torch.empty(0, device="cuda")
quant_func = get_quant(quant_type)
```

**`aiter/jit/core.py`**
```
f"No existing config files found in '{file_path}' "
f"when merging '{merge_name}'."
_FILL_DEFAULTS = {"xbf16": 0, "run_1stage": 0, "ksplit": 0}
all_cols = list(source_pairs[0][1].columns)
```

**`aiter/utility/base_tuner.py`**
```
if len(df_list) > 1:
all_cols = list(df_list[0].columns)
for df in df_list[1:]:
for c in df.columns:
```

**`csrc/ck_gemm_moe_2stages_codegen/gemm_moe_tune.py`**
```
q_type == QuantType.per_1x128
and q_dtype_a == dtypes.fp8
and get_gfx() == "gfx950"
xbf16_csv = kernels_list_csv_1stage.format(
```

**`csrc/py_itfs_cu/asm_fmoe.cu`**
```
bool xquant = (input->dtype() == AITER_DTYPE_bf16);
config_map = xquant ? &cfg_fmoe_bf16_blockscaleBf16_g1u1_silu : &cfg_fmoe_bf16_blockscaleFp8_g1u1_silu;
config_map = xquant ? &cfg_fmoe_bf16_blockscaleBf16_g1u1_gelu : &cfg_fmoe_bf16_blockscaleFp8_g1u1_gelu;
```
