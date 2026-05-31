# Diff summary

- **files changed:** 13
- **lines:** +117 / -49
- **kernel-ish files:** 13

## Files (by churn)

- `op_tests/triton_tests/test_gemm_a8w8_blockscale.py`  (+27/-9)
- `op_tests/triton_tests/test_gemm_a8w8.py`  (+15/-10)
- `op_tests/triton_tests/test_batched_gemm_bf16.py`  (+16/-6)
- `op_tests/triton_tests/test_batched_gemm_a8w8.py`  (+15/-6)
- `op_tests/triton_tests/test_gemm_a16w16.py`  (+13/-5)
- `op_tests/op_benchmarks/triton/bench_gemm_a8w8.py`  (+5/-3)
- `aiter/ops/triton/gemm_a16w16.py`  (+5/-1)
- `op_tests/op_benchmarks/triton/bench_gemm_a8w8_blockscale.py`  (+3/-3)
- `aiter/ops/triton/batched_gemm_a8w8.py`  (+4/-1)
- `aiter/ops/triton/batched_gemm_bf16.py`  (+4/-1)
- `aiter/ops/triton/gemm_a8w8.py`  (+4/-1)
- `aiter/ops/triton/gemm_a8w8_blockscale.py`  (+4/-1)
- `op_tests/op_benchmarks/triton/bench_gemm_a16w16.py`  (+2/-2)

## Key added lines (kernel files)

**`aiter/ops/triton/batched_gemm_a8w8.py`**
```
YQ: Optional[torch.Tensor] = None,
- YQ: Output Matrix Y with shape (B, M, N). If this is none, then it's created by this API and returned as output
if YQ is None:
YQ = torch.empty((B, M, N), dtype=dtype, device=XQ.device)
```

**`aiter/ops/triton/batched_gemm_bf16.py`**
```
YQ: Optional[torch.Tensor] = None,
- YQ: Output Matrix Y with shape (B, M, N). If this is none, then it's created by this API and returned as output
if YQ is None:
YQ = torch.empty((B, M, N), dtype=dtype, device=XQ.device)
```

**`aiter/ops/triton/gemm_a16w16.py`**
```
y: Optional[torch.Tensor] = None,
- dtype: Optional parameter to specifcy bf16 or fp16 datatype. Default is bf16
- Y: Output Matrix Y with shape (M, N). If this is none, then it's created by this API and returned as output
if y is None:
```

**`aiter/ops/triton/gemm_a8w8.py`**
```
y: Optional[torch.Tensor] = None,
- Y: Output Matrix Y with shape (M, K). If this is none, then it's created by this API and returned as output
if y is None:
y = torch.empty((M, N), dtype=dtype, device=x.device)
```

**`aiter/ops/triton/gemm_a8w8_blockscale.py`**
```
y: Optional[torch.Tensor] = None,
- Y: Output Matrix Y with shape (M, K). If this is none, then it's created by this API and returned as output
if y is None:
y = torch.empty((M, N), dtype=dtype, device=x.device)
```
