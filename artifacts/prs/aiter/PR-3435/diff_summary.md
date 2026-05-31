# Diff summary

- **files changed:** 11
- **lines:** +251 / -500
- **kernel-ish files:** 7

## Files (by churn)

- `aiter/ops/flydsl/kernels/mixed_moe_gemm_2stage.py`  (+136/-248)
- `csrc/ck_gemm_moe_2stages_codegen/gemm_moe_tune.py`  (+1/-69)
- `aiter/configs/model_configs/gptoss_fp4_tuned_fmoe.csv`  (+25/-25)
- `aiter/configs/model_configs/gptoss_fp4_untuned_fmoe.csv`  (+25/-25)
- `aiter/fused_moe.py`  (+4/-43)
- `op_tests/test_moe_2stage.py`  (+17/-26)
- `aiter/aot/flydsl/moe.py`  (+15/-24)
- `aiter/configs/model_configs/gptoss_fp8fp4_tuned_fmoe.csv`  (+15/-15)
- `aiter/ops/flydsl/moe_kernels.py`  (+4/-16)
- `aiter/configs/model_configs/gptoss_fp8fp4_untuned_fmoe.csv`  (+8/-8)
- `aiter/jit/core.py`  (+1/-1)

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

**`aiter/ops/flydsl/kernels/mixed_moe_gemm_2stage.py`**
```
model_dim_pad: int = 0,
inter_dim_pad: int = 0,
_inter_dim_valid = inter_dim - inter_dim_pad
_c_idp_sw = arith.constant(2 * inter_dim_pad, index=True)
```

**`aiter/ops/flydsl/moe_kernels.py`**
```
model_dim_pad=model_dim_pad,
inter_dim_pad=inter_dim_pad,
model_dim_pad=model_dim_pad,
inter_dim_pad=inter_dim_pad,
```
