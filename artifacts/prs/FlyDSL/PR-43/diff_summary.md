# Diff summary

- **files changed:** 15
- **lines:** +456 / -854
- **kernel-ish files:** 13

## Files (by churn)

- `tests/kernels/test_gemm.py`  (+0/-705)
- `scripts/run_benchmark.sh`  (+239/-0)
- `kernels/reduce.py`  (+89/-75)
- `kernels/rmsnorm_kernel.py`  (+34/-33)
- `tests/kernels/test_preshuffle_gemm.py`  (+28/-11)
- `kernels/softmax_kernel.py`  (+16/-12)
- `.github/workflows/flydsl.yaml`  (+17/-1)
- `kernels/layernorm_kernel.py`  (+8/-7)
- `tests/kernels/test_quant.py`  (+11/-0)
- `tests/kernels/test_moe_gemm.py`  (+2/-4)
- `flydsl/src/flydsl/dialects/ext/flir.py`  (+4/-1)
- `tests/kernels/test_matrix_trans.py`  (+2/-2)
- `tests/kernels/test_layernorm.py`  (+2/-1)
- `tests/kernels/test_rmsnorm.py`  (+2/-1)
- `tests/kernels/test_softmax.py`  (+2/-1)

## Key added lines (kernel files)

**`flydsl/src/flydsl/dialects/ext/flir.py`**
```
vec_load_op = vector.load(
vec_val = vec_load_op.result if hasattr(vec_load_op, "result") else vec_load_op
vec_val = _unwrap_value(vec_val)
```

**`kernels/layernorm_kernel.py`**
```
c_vecw = flir.const_index(VEC_WIDTH)
thread_offset_base = tid * c_vecw
diff = arith.as_value(arith.subf(x, mean_splat))
norm = arith.as_value(arith.mulf(diff, rstd_splat))
```

**`kernels/reduce.py`**
```
from flydsl.dialects.ext import arith as _arith
vec_val = _arith.as_value(vec_val)
except Exception:
from flydsl.dialects.ext import arith as _arith
```

**`kernels/rmsnorm_kernel.py`**
```
n_float = arith.constant(float(N), type=compute_type)
eps = arith.constant(float(EPS), type=compute_type)
x = (x_e) if dtype_str == "f32" else flir.arith.extf(compute_type, arith.as_value(x_e))
x = (x_e) if dtype_str == "f32" else flir.arith.extf(compute_type, arith.as_value(x_e))
```

**`kernels/softmax_kernel.py`**
```
from flydsl.dialects.ext import flir, arith, gpu
from _mlir.dialects import vector
c_vecw = flir.const_index(VEC_WIDTH)
thread_offset_base = (arith.ArithValue(tid) * VEC_WIDTH).value
```
