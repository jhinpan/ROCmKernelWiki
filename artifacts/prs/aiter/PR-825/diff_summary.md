# Diff summary

- **files changed:** 21
- **lines:** +179 / -142
- **kernel-ish files:** 20

## Files (by churn)

- `op_tests/triton_tests/test_layernorm.py`  (+23/-24)
- `op_tests/triton_tests/test_extend_attention.py`  (+31/-8)
- `op_tests/triton_tests/test_rmsnorm.py`  (+19/-20)
- `op_tests/triton_tests/test_batched_gemm_a8w8.py`  (+13/-12)
- `op_tests/triton_tests/test_prefill_attention.py`  (+12/-10)
- `op_tests/triton_tests/test_chunked_pa_prefill.py`  (+12/-9)
- `op_tests/triton_tests/test_gemm_a16w16.py`  (+10/-11)
- `op_tests/triton_tests/test_batched_gemm_bf16.py`  (+11/-8)
- `op_tests/triton_tests/test_pa_prefill.py`  (+12/-7)
- `op_tests/triton_tests/test_batched_gemm_afp4wfp4.py`  (+8/-7)
- `op_tests/triton_tests/test_gemm_a16w16_gated.py`  (+6/-7)
- `op_tests/triton_tests/README.md`  (+9/-0)
- `op_tests/triton_tests/test_gemm_a8w8.py`  (+3/-4)
- `op_tests/op_benchmarks/triton/bench_extend_attention.py`  (+3/-2)
- `op_tests/triton_tests/test_gemm_a8w8_per_token_scale.py`  (+2/-3)

## Key added lines (kernel files)

**`op_tests/triton_tests/test_batched_gemm_a8w8.py`**
```
x = torch.randint(-20, 20, (B, M, K), dtype=torch.int8, device="cuda")
x = torch.randint(-20, 20, (B, K, M), dtype=torch.int8, device="cuda").permute(
weight = torch.randint(-20, 20, (B, N, K), dtype=torch.int8, device="cuda")
weight = torch.randint(
```

**`op_tests/triton_tests/test_batched_gemm_afp4wfp4.py`**
```
x_low = torch.randint(0, 16, (B, M, K // 2), dtype=torch.uint8, device="cuda")
x_high = torch.randint(0, 16, (B, M, K // 2), dtype=torch.uint8, device="cuda")
x_low = torch.randint(
0, 16, (B, K // 2, M), dtype=torch.uint8, device="cuda"
```

**`op_tests/triton_tests/test_batched_gemm_bf16.py`**
```
x = torch.randint(-20, 20, (B, M, K), dtype=dtype, device="cuda")
x = torch.randint(-20, 20, (B, K, M), dtype=dtype, device="cuda").permute(
weight = torch.randint(-20, 20, (B, N, K), dtype=dtype, device="cuda")
weight = torch.randint(-20, 20, (B, K, N), dtype=dtype, device="cuda").permute(
```

**`op_tests/triton_tests/test_chunked_pa_prefill.py`**
```
def _get_alibi_slopes(total_num_heads: int, device: torch.device) -> torch.Tensor:
device=device,
powers = torch.arange(1, 1 + closest_power_of_2, dtype=torch.int32, device=device)
device=device,
```

**`op_tests/triton_tests/test_extend_attention.py`**
```
max_extend_length + 1,
dtype=torch.int32,
device=device,
seqlens_prefix = torch.full((B,), prefix_length, device=device)
```
