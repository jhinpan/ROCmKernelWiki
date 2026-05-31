# Diff summary

- **files changed:** 28
- **lines:** +185 / -84
- **kernel-ish files:** 5

## Files (by churn)

- `gradlib/gradlib/GemmTuner.py`  (+103/-25)
- `csrc/py_itfs_cu/asm_gemm_a16w16.cu`  (+37/-8)
- `op_tests/test_gemm_a16w16.py`  (+19/-25)
- `hsa/gfx950/bf16gemm/bf16gemm_fp32bf16.csv`  (+13/-13)
- `hsa/gfx942/bf16gemm/bf16gemm_fp32bf16.csv`  (+11/-10)
- `aiter/configs/bf16_untuned_gemm.csv`  (+1/-1)
- `gradlib/gradlib/gemm_tuner.py`  (+1/-1)
- `aiter/tuned_gemm.py`  (+0/-1)
- `hsa/gfx942/bf16gemm/bf16gemm_fp32bf16_tn_128x64_bshuffle_splitk.co`  (+0/-0)
- `hsa/gfx942/bf16gemm/bf16gemm_fp32bf16_tn_160x64_bshuffle_splitk.co`  (+0/-0)
- `hsa/gfx942/bf16gemm/bf16gemm_fp32bf16_tn_32x64_bshuffle_splitk.co`  (+0/-0)
- `hsa/gfx942/bf16gemm/bf16gemm_fp32bf16_tn_32x64_pf3_splitk.co`  (+0/-0)
- `hsa/gfx942/bf16gemm/bf16gemm_fp32bf16_tn_48x64_bshuffle_splitk.co`  (+0/-0)
- `hsa/gfx942/bf16gemm/bf16gemm_fp32bf16_tn_48x64_pf3_splitk.co`  (+0/-0)
- `hsa/gfx942/bf16gemm/bf16gemm_fp32bf16_tn_64x64_bshuffle_splitk.co`  (+0/-0)

## Key added lines (kernel files)

**`csrc/py_itfs_cu/asm_gemm_a16w16.cu`**
```
void *ptr_D;
void *ptr_C;
void *ptr_A;
void *ptr_B;
```

**`gradlib/gradlib/GemmTuner.py`**
```
from aiter.ops.triton.gemm_a16w16 import gemm_a16w16 as triton_gemm_a16w16
def run_triton_gemm_bf16(input, weight, bias=None, otype=dtypes.bf16):
return triton_gemm_a16w16(input, weight, bias=bias, dtype=otype)
scale = scale_half if scaleAB else None
```

**`gradlib/gradlib/gemm_tuner.py`**
```
if process.exitcode > 1:
```

**`op_tests/test_gemm_a16w16.py`**
```
a, avg_a = run_torch(x, weight, bias, otype, scaleA, scaleB)
b, avg_b = run_gemm_b(x, weight, bias, otype, scaleA, scaleB)
c, avg_c = aiter_hip_bpreshuffle(x, weight_bpreshuffle, None, None, otype)
and (otype == dtypes.fp32 or otype == dtypes.bf16)
```
