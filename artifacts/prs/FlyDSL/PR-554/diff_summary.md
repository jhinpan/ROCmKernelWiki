# Diff summary

- **files changed:** 11
- **lines:** +387 / -62
- **kernel-ish files:** 11

## Files (by churn)

- `python/flydsl/compiler/jit_argument.py`  (+116/-7)
- `tests/unit/test_tensor_cache_signature.py`  (+110/-0)
- `kernels/mixed_moe_gemm_2stage.py`  (+37/-5)
- `python/flydsl/compiler/jit_function.py`  (+27/-15)
- `kernels/moe_blockscale_2stage.py`  (+24/-11)
- `kernels/moe_gemm_2stage.py`  (+21/-10)
- `python/flydsl/expr/rocdl/universal.py`  (+24/-4)
- `kernels/fp8_gemm_utils.py`  (+11/-5)
- `kernels/preshuffle_gemm.py`  (+11/-1)
- `kernels/fp8_gemm_4wave.py`  (+3/-2)
- `kernels/fp8_gemm_8wave.py`  (+3/-2)

## Key added lines (kernel files)

**`kernels/fp8_gemm_4wave.py`**
```
gA = make_fp8_buffer_tensor(A, F8_IR_t, num_records_bytes=M * K)
gB = make_fp8_buffer_tensor(B_T, F8_IR_t, num_records_bytes=N * K)
```

**`kernels/fp8_gemm_8wave.py`**
```
gA = make_fp8_buffer_tensor(A, F8_IR_t, num_records_bytes=M * K)
gB = make_fp8_buffer_tensor(B_T, F8_IR_t, num_records_bytes=N * K)
```

**`kernels/fp8_gemm_utils.py`**
```
def make_fp8_buffer_tensor(arg_i8, fp8_ir_t, num_records_bytes):
t_i8 = fx.rocdl.make_buffer_tensor(arg_i8, max_size=False, num_records_bytes=num_records_bytes)
c_nbytes = c_rows * c_cols * 2  # BFloat16 = 2 bytes
sa_nbytes = c_rows * 4  # Float32 row-wise scale
```

**`kernels/mixed_moe_gemm_2stage.py`**
```
w_nbytes_s1 = (
(experts * (2 * inter_dim) * model_dim) // 2
if is_f4_b
else (experts * (2 * inter_dim) * model_dim * b_elem_bytes)
```

**`kernels/moe_blockscale_2stage.py`**
```
nblk_n_w1 = (2 * inter_dim) // 128  # N-blocks in W1 (ScaleBlockN=128)
sw_nbytes = experts * nblk_n_w1 * nblk_k_w1 * 4
w_nbytes = (
(experts * (2 * inter_dim) * model_dim) // 2
```
