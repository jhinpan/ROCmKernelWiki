# Diff summary

- **files changed:** 123
- **lines:** +15 / -14
- **kernel-ish files:** 4

## Files (by churn)

- `aiter/ops/gemm_op_a8w8.py`  (+8/-8)
- `op_tests/test_gemm_a8w8_blockscale_mi350.py`  (+2/-2)
- `aiter/ops/triton/utils/_triton/arch_info.py`  (+2/-1)
- `aiter/ops/triton/configs/gemm/gfx950-BATCHED_GEMM-AFP4WFP4.json`  (+1/-1)
- `aiter/ops/triton/configs/gemm/gfx950-BATCHED_GEMM_PREQUANT-AFP4WFP4.json`  (+1/-1)
- `op_tests/triton_tests/test_moe.py`  (+1/-1)
- `aiter/ops/triton/configs/gemm/gfx950-BATCHED_GEMM-A16W16.json`  (+0/-0)
- `aiter/ops/triton/configs/gemm/gfx950-BATCHED_GEMM-A8W8-A_PER_TOKEN_GROUP_PREQUANT_W_PER_BATCHED_TENSOR_QUANT-N=128-K=512.json`  (+0/-0)
- `aiter/ops/triton/configs/gemm/gfx950-BATCHED_GEMM-A8W8-A_PER_TOKEN_GROUP_PREQUANT_W_PER_BATCHED_TENSOR_QUANT-N=512-K=128.json`  (+0/-0)
- `aiter/ops/triton/configs/gemm/gfx950-BATCHED_GEMM-A8W8-A_PER_TOKEN_GROUP_PREQUANT_W_PER_BATCHED_TENSOR_QUANT.json`  (+0/-0)
- `aiter/ops/triton/configs/gemm/gfx950-BATCHED_GEMM-A8W8.json`  (+0/-0)
- `aiter/ops/triton/configs/gemm/gfx950-BATCHED_GEMM-AFP4WFP4-N=128-K=512.json`  (+0/-0)
- `aiter/ops/triton/configs/gemm/gfx950-BATCHED_GEMM-AFP4WFP4-N=512-K=128.json`  (+0/-0)
- `aiter/ops/triton/configs/gemm/gfx950-BATCHED_GEMM_PREQUANT-AFP4WFP4-N=128-K=512.json`  (+0/-0)
- `aiter/ops/triton/configs/gemm/gfx950-BATCHED_GEMM_PREQUANT-AFP4WFP4-N=512-K=128.json`  (+0/-0)

## Key added lines (kernel files)

**`aiter/ops/gemm_op_a8w8.py`**
```
def gen_gfx950_a8w8_blockscale_asm_fake_tensors(
"module_gemm_gfx950_a8w8_blockscale_asm",
fc_name="gfx950_a8w8_blockscale_asm",
gen_fake=gen_gfx950_a8w8_blockscale_asm_fake_tensors,
```

**`aiter/ops/triton/utils/_triton/arch_info.py`**
```
"gfx950": "gfx950",
```

**`op_tests/test_gemm_a8w8_blockscale_mi350.py`**
```
def test_gemm_asm_gfx950(dtype, m, n, k):
test_gemm_asm_gfx950(dtype, m, n, k)
```

**`op_tests/triton_tests/test_moe.py`**
```
if dev == "gfx950":
```
