# Diff summary

- **files changed:** 9
- **lines:** +47 / -36
- **kernel-ish files:** 7

## Files (by churn)

- `aiter/fused_moe.py`  (+14/-15)
- `hsa/gfx942/fmoe_2stages/tune.py`  (+18/-11)
- `op_tests/test_moe_ep.py`  (+7/-5)
- `csrc/ck_gemm_moe_2stages_codegen/gen_instances.py`  (+3/-0)
- `aiter/configs/tuned_fmoe.csv`  (+1/-1)
- `aiter/configs/untuned_fmoe.csv`  (+1/-1)
- `aiter/fused_moe_bf16_asm.py`  (+1/-1)
- `aiter/ops/moe_op.py`  (+1/-1)
- `csrc/ck_gemm_moe_2stages_codegen/gemm_moe_ck2stages_common.py`  (+1/-1)

## Key added lines (kernel files)

**`aiter/fused_moe.py`**
```
run_1stage = cfg.get("run_1stage", False)
if "ck2stages" in kernelName1 or q_dtype_w in [
dtypes.bf16,
dtypes.fp16,
```

**`aiter/fused_moe_bf16_asm.py`**
```
num_valid_ids = torch.empty((2), dtype=dtypes.i32, device=device)
```

**`aiter/ops/moe_op.py`**
```
QuantType.per_1x128 if quant_type == QuantType.per_128x128 else quant_type
```

**`csrc/ck_gemm_moe_2stages_codegen/gemm_moe_ck2stages_common.py`**
```
QuantType_list = ["per_1x128", "per_1x32"]
```

**`csrc/ck_gemm_moe_2stages_codegen/gen_instances.py`**
```
args.quant_type = (
"per_1x128" if args.quant_type == "per_128x128" else args.quant_type
```
