# Diff summary

- **files changed:** 14
- **lines:** +2773 / -406
- **kernel-ish files:** 10

## Files (by churn)

- `aiter/ops/flydsl/kernels/moe_gemm_2stage.py`  (+1261/-326)
- `csrc/ck_gemm_moe_2stages_codegen/gemm_moe_tune.py`  (+322/-20)
- `aiter/ops/flydsl/kernels/mfma_preshuffle_pipeline.py`  (+290/-31)
- `aiter/ops/flydsl/kernels/swiglu_and_mul.py`  (+206/-0)
- `aiter/fused_moe.py`  (+163/-13)
- `aiter/ops/flydsl/moe_kernels.py`  (+124/-6)
- `aiter/configs/model_configs/kimik2_i4_untuned_fmoe.csv`  (+85/-0)
- `aiter/configs/model_configs/kimik2_i4_tuned_fmoe.csv`  (+83/-0)
- `op_tests/test_moe_2stage.py`  (+67/-8)
- `aiter/configs/model_configs/kimik2_fp8fp4_tuned_fmoe.csv`  (+62/-0)
- `aiter/ops/shuffle.py`  (+46/-2)
- `aiter/configs/model_configs/kimik2_fp8fp4_untuned_fmoe.csv`  (+31/-0)
- `aiter/ops/quant.py`  (+23/-0)
- `aiter/jit/utils/moe_recipes.py`  (+10/-0)

## Key added lines (kernel files)

**`aiter/fused_moe.py`**
```
bf16_fp8_bound = int(os.environ.get("AITER_BF16_FP8_BOUND", "512"))
if quant_type == QuantType.per_1x32 and q_dtype_w == dtypes.i4x2:
q_dtype_a = dtypes.bf16
elif quant_type == QuantType.per_1x32:
```

**`aiter/jit/utils/moe_recipes.py`**
```
kn1 = (row.get("kernelName1") or "").strip()
kn2 = (row.get("kernelName2") or "").strip()
if kn1.startswith("flydsl_") or kn2.startswith("flydsl_"):
if quant_type == "per_1x32" and a_dtype == "b16" and b_dtype == "torch.int4":
```

**`aiter/ops/flydsl/kernels/mfma_preshuffle_pipeline.py`**
```
def lds_row_major_idx(row, col, row_stride, base=None):
"""Linearize a 2D LDS coordinate with explicit index arithmetic."""
idx = row * row_stride + col
return idx if base is None else idx + base
```

**`aiter/ops/flydsl/kernels/moe_gemm_2stage.py`**
```
from flydsl._mlir.dialects import llvm, scf
load_b_raw_w4a16_groupwise,
extract_bf16_scale,
scale_is_bf16: bool = False,
```

**`aiter/ops/flydsl/kernels/swiglu_and_mul.py`**
```
"""Fused swiglu_and_mul kernel for interleaved (N0, 2, NLane) layout (FlyDSL).
Input layout (from cktile split_k with a16w4 interleave preshuffle):
For each row of inter_dim*2 columns, data is arranged as:
[gate_block0(NLane), up_block0(NLane), gate_block1(NLane), up_block1(NLane), ...]
```
