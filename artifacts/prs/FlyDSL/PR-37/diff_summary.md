# Diff summary

- **files changed:** 11
- **lines:** +1419 / -407
- **kernel-ish files:** 11

## Files (by churn)

- `kernels/moe_gemm_2stage.py`  (+711/-238)
- `kernels/mfma_epilogues.py`  (+261/-0)
- `kernels/preshuffle_gemm.py`  (+118/-26)
- `tests/kernels/test_moe_gemm.py`  (+110/-14)
- `kernels/mfma_preshuffle_pipeline.py`  (+104/-15)
- `tests/kernels/test_gemm.py`  (+34/-39)
- `kernels/layernorm_kernel.py`  (+32/-32)
- `kernels/rmsnorm_kernel.py`  (+30/-30)
- `kernels/softmax_kernel.py`  (+9/-10)
- `flydsl/src/flydsl/dialects/ext/arith.py`  (+9/-2)
- `flydsl/src/flydsl/dialects/ext/flir.py`  (+1/-1)

## Key added lines (kernel files)

**`flydsl/src/flydsl/dialects/ext/arith.py`**
```
if op == "xor":
op_name = "XOrI"
op_name = op.capitalize() + "I"  # AndI, OrI
for t in [F32Type, F64Type, IndexType, IntegerType, VectorType]:
```

**`flydsl/src/flydsl/dialects/ext/flir.py`**
```
return vector.BitCastOp(vec_elem_ty, _unwrap_value(i32_vec)).result
```

**`kernels/layernorm_kernel.py`**
```
x2 = x * x
thread_sum = thread_sum + red
thread_sumsq = thread_sumsq + red2
mean = sum_val * inv_n
```

**`kernels/mfma_epilogues.py`**
```
"""Reusable epilogue helpers for MFMA 16x16-based kernels.
This module provides:
- `mfma_epilog(...)`
A single entrypoint that dispatches to either the default row-epilogue or the
```

**`kernels/mfma_preshuffle_pipeline.py`**
```
c_k0 = c_k / c64
n0 = c_n / c16
stride_klane = c16 * stride_nlane
stride_k0 = c4 * stride_klane
```
