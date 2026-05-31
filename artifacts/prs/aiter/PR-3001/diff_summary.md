# Diff summary

- **files changed:** 11
- **lines:** +556 / -148
- **kernel-ish files:** 4

## Files (by churn)

- `csrc/ck_gemm_moe_2stages_codegen/gemm_moe_tune.py`  (+303/-16)
- `aiter/configs/model_configs/qwen3next_80b_fp8_tuned_fmoe.csv`  (+119/-97)
- `hsa/codegen.py`  (+47/-18)
- `aiter/fused_moe.py`  (+51/-2)
- `csrc/py_itfs_cu/asm_fmoe.cu`  (+22/-5)
- `hsa/gfx950/fmoe/gelu/fmoe_bf16_blockscaleBf16_g1u1_gelu.csv`  (+7/-5)
- `hsa/gfx950/fmoe/silu/fmoe_bf16_blockscaleBf16_g1u1_silu.csv`  (+7/-5)
- `hsa/gfx950/fmoe/gelu/fmoe_bf16_blockscaleBf16_g1u1_flat_vs_gelu_1x128.co`  (+0/-0)
- `hsa/gfx950/fmoe/gelu/fmoe_bf16_blockscaleBf16_g1u1_flat_vs_gelu_1x256.co`  (+0/-0)
- `hsa/gfx950/fmoe/silu/fmoe_bf16_blockscaleBf16_g1u1_flat_vs_silu_1x128.co`  (+0/-0)
- `hsa/gfx950/fmoe/silu/fmoe_bf16_blockscaleBf16_g1u1_flat_vs_silu_1x256.co`  (+0/-0)

## Key added lines (kernel files)

**`aiter/fused_moe.py`**
```
def _moe_prepare_unsorted_input(topk_ids, topk_weights, model_dim, moebuf_dtype):
device = topk_ids.device
M = topk_ids.shape[0]
elem_size = torch.empty(0, dtype=moebuf_dtype).element_size()
```

**`csrc/ck_gemm_moe_2stages_codegen/gemm_moe_tune.py`**
```
def _manifest_flat_by_kernel(df: pd.DataFrame) -> dict:
"""Map ``knl_name`` -> 0/1 when the manifest has a ``flat`` column.
If the column is absent, every kernel is treated as non-FLAT (equivalent
to all zeros). Only manifests that include FLAT 1-stage asm variants need
```

**`csrc/py_itfs_cu/asm_fmoe.cu`**
```
bool is_flat_dispatch = false;
uint32_t num_persistent_tgs = 0,
bool is_flat_dispatch       = false) : kernel(name, hsaco)
this->is_flat_dispatch   = is_flat_dispatch;
```

**`hsa/codegen.py`**
```
cfg_entries = []
cfg_entries.append((cfgname, relpath, combine_df))
if cfg_entries:
required_columns = {"knl_name", "co_name", "arch"}
```
