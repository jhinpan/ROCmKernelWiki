# Diff summary

- **files changed:** 9
- **lines:** +273 / -113
- **kernel-ish files:** 5

## Files (by churn)

- `op_tests/test_moe_2stage.py`  (+60/-23)
- `csrc/ck_gemm_moe_2stages_codegen/gemm_moe_tune.py`  (+63/-0)
- `aiter/fused_moe.py`  (+51/-4)
- `aiter/configs/model_configs/gptoss_fp4_tuned_fmoe.csv`  (+25/-25)
- `aiter/configs/model_configs/gptoss_fp4_untuned_fmoe.csv`  (+25/-25)
- `aiter/configs/model_configs/gptoss_fp8fp4_tuned_fmoe.csv`  (+15/-15)
- `aiter/aot/flydsl/moe.py`  (+17/-12)
- `aiter/configs/model_configs/gptoss_fp8fp4_untuned_fmoe.csv`  (+8/-8)
- `aiter/jit/core.py`  (+9/-1)

## Key added lines (kernel files)

**`aiter/aot/flydsl/moe.py`**
```
hidden_pad = int(row.get("hidden_pad", "0") or "0")
intermediate_pad = int(row.get("intermediate_pad", "0") or "0")
enable_bias_options = [str(row.get("bias", "")).strip() == "True"]
"hidden_pad": hidden_pad,
```

**`aiter/fused_moe.py`**
```
bias=(bias1 is not None or bias2 is not None),
bias=False,
"hidden_pad",
"intermediate_pad",
```

**`aiter/jit/core.py`**
```
_FILL_DEFAULTS = {
"xbf16": 0,
"run_1stage": 0,
"ksplit": 0,
```

**`csrc/ck_gemm_moe_2stages_codegen/gemm_moe_tune.py`**
```
hidden_pad,
intermediate_pad,
hidden_pad,
intermediate_pad,
```

**`op_tests/test_moe_2stage.py`**
```
bias=False,
0 <= hidden_pad < model_dim
), f"invalid hidden_pad={hidden_pad} for model_dim={model_dim}"
0 <= intermediate_pad < inter_dim
```
