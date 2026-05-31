# Diff summary

- **files changed:** 32
- **lines:** +67 / -17
- **kernel-ish files:** 32

## Files (by churn)

- `op_tests/triton_tests/attention/test_pa_decode.py`  (+8/-3)
- `op_tests/triton_tests/quant/test_fused_fp8_quant.py`  (+9/-2)
- `op_tests/triton_tests/attention/test_mha.py`  (+6/-4)
- `op_tests/triton_tests/fusions/test_fused_kv_cache.py`  (+7/-0)
- `op_tests/triton_tests/moe/test_moe_mx.py`  (+3/-3)
- `op_tests/triton_tests/gemm/basic/test_gemm_a8wfp4.py`  (+3/-2)
- `op_tests/triton_tests/attention/test_fav3_sage.py`  (+3/-0)
- `op_tests/triton_tests/fusions/test_fused_bmm_rope_kv_cache.py`  (+2/-1)
- `op_tests/triton_tests/attention/test_fp8_mqa_logits.py`  (+1/-1)
- `op_tests/triton_tests/gemm/feed_forward/test_ff_a16w16.py`  (+2/-0)
- `op_tests/triton_tests/gemm/feed_forward/test_ff_a16w16_fused.py`  (+2/-0)
- `op_tests/triton_tests/test_gated_delta_rule.py`  (+1/-1)
- `op_tests/triton_tests/attention/test_la_paged.py`  (+1/-0)
- `op_tests/triton_tests/fusions/test_fused_qk_concat.py`  (+1/-0)
- `op_tests/triton_tests/gemm/basic/test_gemm_a16w16.py`  (+1/-0)

## Key added lines (kernel files)

**`op_tests/triton_tests/attention/test_fav3_sage.py`**
```
torch.manual_seed(20)
torch.manual_seed(20)
```

**`op_tests/triton_tests/attention/test_fp8_mqa_logits.py`**
```
torch.manual_seed(0)
```

**`op_tests/triton_tests/attention/test_la_paged.py`**
```
random.seed(20)
```

**`op_tests/triton_tests/attention/test_mha.py`**
```
torch.cuda.empty_cache()
torch.manual_seed(20)
torch.cuda.empty_cache()
torch.manual_seed(20)
```

**`op_tests/triton_tests/attention/test_pa_decode.py`**
```
torch.manual_seed(random_seed)
torch.cuda.empty_cache()  # Helps avoid hangs in large tests
torch.cuda.empty_cache()  # Helps avoid hangs in large tests
torch.set_printoptions(precision=5, threshold=10000)
```
