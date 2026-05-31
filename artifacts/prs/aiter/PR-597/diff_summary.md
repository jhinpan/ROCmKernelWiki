# Diff summary

- **files changed:** 22
- **lines:** +137 / -79
- **kernel-ish files:** 22

## Files (by churn)

- `op_tests/triton_tests/test_batched_gemm_afp4wfp4.py`  (+27/-10)
- `op_tests/triton_tests/test_batched_gemm_afp4wfp4_pre_quant.py`  (+16/-7)
- `aiter/ops/triton/gemm_a8w8_blockscale.py`  (+7/-8)
- `aiter/ops/triton/batched_gemm_a8w8.py`  (+8/-6)
- `aiter/ops/triton/batched_gemm_bf16.py`  (+8/-6)
- `aiter/ops/triton/gemm_afp4wfp4.py`  (+10/-4)
- `aiter/ops/triton/gemm_a8w8.py`  (+6/-6)
- `aiter/ops/triton/gemm_a8wfp4.py`  (+6/-5)
- `aiter/ops/triton/gemm_afp4wfp4_pre_quant_atomic.py`  (+7/-4)
- `aiter/ops/triton/utils/pid_preprocessing.py`  (+10/-1)
- `op_tests/triton_tests/test_gemm_afp4wfp4_pre_quant_atomic.py`  (+5/-5)
- `op_tests/triton_tests/test_gemm_a16w16.py`  (+6/-3)
- `op_tests/triton_tests/test_gemm_afp4wfp4.py`  (+4/-5)
- `aiter/ops/triton/gemm_a16w16.py`  (+4/-3)
- `aiter/ops/triton/batched_gemm_afp4wfp4.py`  (+3/-2)

## Key added lines (kernel files)

**`aiter/ops/triton/batched_gemm_a8w8.py`**
```
tl.assume(pid_m >= 0)
tl.assume(pid_n >= 0)
batch_id = tl.cast(batch_id, tl.int64)
stride_ab = tl.cast(stride_ab, tl.int64)
```

**`aiter/ops/triton/batched_gemm_afp4wfp4.py`**
```
- W: Matrix W with shape (B, N, K).
- Y: The output matrix with shape (B, M, N).
w = w.transpose(1, 2)
```

**`aiter/ops/triton/batched_gemm_afp4wfp4_pre_quant.py`**
```
- W: Matrix W with shape (B, N, K).
Bw, N, K = w.shape
w = w.transpose(1, 2)
```

**`aiter/ops/triton/batched_gemm_bf16.py`**
```
tl.assume(pid_m >= 0)
tl.assume(pid_n >= 0)
batch_id = tl.cast(batch_id, tl.int64)
stride_ab = tl.cast(stride_ab, tl.int64)
```

**`aiter/ops/triton/gemm_a16w16.py`**
```
tl.assume(pid_m >= 0)
tl.assume(pid_n >= 0)
N, K = w.shape
```
