# Diff summary

- **files changed:** 28
- **lines:** +67 / -18
- **kernel-ish files:** 28

## Files (by churn)

- `op_tests/triton_tests/test_moe_mx.py`  (+4/-17)
- `op_tests/triton_tests/test_gemm_afp4wfp4_pre_quant_atomic.py`  (+4/-1)
- `op_tests/triton_tests/test_moe.py`  (+5/-0)
- `op_tests/triton_tests/test_activation.py`  (+4/-0)
- `op_tests/triton_tests/test_fused_mxfp4_quant.py`  (+4/-0)
- `op_tests/triton_tests/test_gemm_a16w16.py`  (+4/-0)
- `op_tests/triton_tests/test_gemm_a8w8.py`  (+3/-0)
- `op_tests/triton_tests/test_mla_decode_rope.py`  (+3/-0)
- `op_tests/triton_tests/test_pa_prefill.py`  (+3/-0)
- `op_tests/triton_tests/test_batched_gemm_a8w8.py`  (+2/-0)
- `op_tests/triton_tests/test_batched_gemm_afp4wfp4.py`  (+2/-0)
- `op_tests/triton_tests/test_batched_gemm_afp4wfp4_pre_quant.py`  (+2/-0)
- `op_tests/triton_tests/test_batched_gemm_bf16.py`  (+2/-0)
- `op_tests/triton_tests/test_extend_attention.py`  (+2/-0)
- `op_tests/triton_tests/test_fused_mul_add.py`  (+2/-0)

## Key added lines (kernel files)

**`op_tests/triton_tests/test_activation.py`**
```
torch.cuda.empty_cache()  # Helps avoid hangs in large tests
```

**`op_tests/triton_tests/test_batched_gemm_a8w8.py`**
```
torch.cuda.empty_cache()  # Helps avoid hangs in large tests
```

**`op_tests/triton_tests/test_batched_gemm_afp4wfp4.py`**
```
torch.cuda.empty_cache()  # Helps avoid hangs in large tests
```

**`op_tests/triton_tests/test_batched_gemm_afp4wfp4_pre_quant.py`**
```
torch.cuda.empty_cache()  # Helps avoid hangs in large tests
```

**`op_tests/triton_tests/test_batched_gemm_bf16.py`**
```
torch.cuda.empty_cache()  # Helps avoid hangs in large tests
```
