# Diff summary

- **files changed:** 11
- **lines:** +500 / -251
- **kernel-ish files:** 7

## Files (by churn)

- `aiter/ops/flydsl/kernels/mixed_moe_gemm_2stage.py`  (+248/-136)
- `csrc/ck_gemm_moe_2stages_codegen/gemm_moe_tune.py`  (+69/-1)
- `aiter/configs/model_configs/gptoss_fp4_tuned_fmoe.csv`  (+25/-25)
- `aiter/configs/model_configs/gptoss_fp4_untuned_fmoe.csv`  (+25/-25)
- `aiter/fused_moe.py`  (+43/-4)
- `op_tests/test_moe_2stage.py`  (+26/-17)
- `aiter/aot/flydsl/moe.py`  (+24/-15)
- `aiter/configs/model_configs/gptoss_fp8fp4_tuned_fmoe.csv`  (+15/-15)
- `aiter/ops/flydsl/moe_kernels.py`  (+16/-4)
- `aiter/configs/model_configs/gptoss_fp8fp4_untuned_fmoe.csv`  (+8/-8)
- `aiter/jit/core.py`  (+1/-1)

## Key added lines (kernel files)

**`aiter/aot/flydsl/moe.py`**
```
def _parse_bool(value: str) -> bool:
value = value.strip()
if value in ("True", "1"):
return True
```

**`aiter/fused_moe.py`**
```
bias=(bias1 is not None or bias2 is not None),
bias=False,
def _normalize_lookup_cols(df):
if "bias" in df.columns:
```

**`aiter/jit/core.py`**
```
_FILL_DEFAULTS = {"xbf16": 0, "run_1stage": 0, "ksplit": 0, "bias": False}
```

**`aiter/ops/flydsl/kernels/mixed_moe_gemm_2stage.py`**
```
gate_only = gate_mode is GateMode.GATE_ONLY
i32_model_dim_pad_in: fx.Int32,
i32_inter_dim_pad_in: fx.Int32,
model_dim_pad_idx = arith.index_cast(
```

**`aiter/ops/flydsl/moe_kernels.py`**
```
model_dim_pad=0,
inter_dim_pad=0,
model_dim_pad,
inter_dim_pad,
```
