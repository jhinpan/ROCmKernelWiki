# Diff summary

- **files changed:** 9
- **lines:** +113 / -273
- **kernel-ish files:** 5

## Files (by churn)

- `op_tests/test_moe_2stage.py`  (+23/-60)
- `csrc/ck_gemm_moe_2stages_codegen/gemm_moe_tune.py`  (+0/-63)
- `aiter/fused_moe.py`  (+4/-51)
- `aiter/configs/model_configs/gptoss_fp4_tuned_fmoe.csv`  (+25/-25)
- `aiter/configs/model_configs/gptoss_fp4_untuned_fmoe.csv`  (+25/-25)
- `aiter/configs/model_configs/gptoss_fp8fp4_tuned_fmoe.csv`  (+15/-15)
- `aiter/aot/flydsl/moe.py`  (+12/-17)
- `aiter/configs/model_configs/gptoss_fp8fp4_untuned_fmoe.csv`  (+8/-8)
- `aiter/jit/core.py`  (+1/-9)

## Key added lines (kernel files)

**`aiter/aot/flydsl/moe.py`**
```
q_type = row.get("q_type", "")
dtype = row.get("dtype", "")
q_dtype_w = row.get("q_dtype_w", "")
bias_supported = (
```

**`aiter/fused_moe.py`**
```
"token,model_dim,inter_dim,expert,topk,act_type,dtype,q_dtype_a,q_dtype_w,q_type,use_g1u1,doweight_stage1"
f"\n{token},{model_dim},{inter_dim},{expert},{topk},{activation},{dtype},{q_dtype_a},{q_dtype_ws},{q_type},{int(use_g1u1
```

**`aiter/jit/core.py`**
```
_FILL_DEFAULTS = {"xbf16": 0, "run_1stage": 0, "ksplit": 0}
```

**`op_tests/test_moe_2stage.py`**
```
exp_bias1 = torch.clamp(torch.randn((E, inter_dim * 2), dtype=dtype), -1.0, 1.0)
exp_bias1 = torch.clamp(torch.randn((E * inter_dim), dtype=dtype), -1.0, 1.0)
exp_bias2 = torch.clamp(torch.randn((E, model_dim), dtype=dtype), -1.0, 1.0)
qType == aiter.QuantType.per_1x32
```
