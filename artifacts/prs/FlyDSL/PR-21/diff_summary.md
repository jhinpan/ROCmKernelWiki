# Diff summary

- **files changed:** 18
- **lines:** +1231 / -555
- **kernel-ish files:** 18

## Files (by churn)

- `python/rocdsl/dialects/ext/python_control_flow.py`  (+544/-49)
- `python/rocdsl/dialects/ext/scf.py`  (+280/-12)
- `tests/benchmark/vec_add.py`  (+109/-103)
- `tests/benchmark/matrix_transpose.py`  (+93/-92)
- `tests/benchmark/per_token_quant_benchmark.py`  (+28/-52)
- `python/examples/reduce.py`  (+33/-44)
- `tests/python/gpu/test_mfma_gemm_fp8_rocir_preshuffle.py`  (+28/-43)
- `tests/python/gpu/test_mfma_gemm_fp8_rocir.py`  (+23/-25)
- `python/examples/softmax_kernel.py`  (+18/-21)
- `python/examples/layernorm_kernel.py`  (+14/-23)
- `python/examples/rmsnorm_kernel.py`  (+18/-19)
- `tests/python/gpu/test_gpu_with_rocir_coords.py`  (+10/-25)
- `tests/python/gpu/test_gpu_rocdsl.py`  (+9/-16)
- `tests/python/gpu/test_shared_working.py`  (+4/-18)
- `python/rocdsl/dialects/ext/rocir.py`  (+13/-1)

## Key added lines (kernel files)

**`python/examples/layernorm_kernel.py`**
```
from rocdsl.dialects.ext.python_control_flow import range_constexpr
for base_idx_int in range_constexpr(0, N, BLOCK_THREADS * VEC_WIDTH):
for k in range_constexpr(VEC_WIDTH):
if is_valid:
```

**`python/examples/reduce.py`**
```
from rocdsl.dialects.ext.python_control_flow import lower_range_for_loops as _lower_range_for_loops
if is_lane0:
if is_wave0:
lane_idx = rocir.arith.IndexCastOp(T.index(), unwrap(lane_i32)).result
```

**`python/examples/rmsnorm_kernel.py`**
```
from rocdsl.dialects.ext.python_control_flow import range_constexpr
for base_idx_int in range_constexpr(0, N, BLOCK_THREADS * VEC_WIDTH):
for k in range_constexpr(VEC_WIDTH):
if is_valid:
```

**`python/examples/softmax_kernel.py`**
```
from rocdsl.dialects.ext.python_control_flow import range_constexpr
for base_idx_int in range_constexpr(0, N, step):
for k in range_constexpr(VEC_WIDTH):
c_last = rocir.const_index(N - 1)
```

**`python/rocdsl/dialects/ext/gpu.py`**
```
Enables lowering of `for i in range(...)` into `scf.for`.
```
