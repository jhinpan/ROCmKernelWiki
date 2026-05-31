# Diff summary

- **files changed:** 13
- **lines:** +221 / -152
- **kernel-ish files:** 13

## Files (by churn)

- `op_tests/triton_tests/attention/test_mha.py`  (+164/-61)
- `op_tests/triton_tests/attention/test_mha_with_sink.py`  (+14/-18)
- `op_tests/triton_tests/attention/test_flash_attn_kvcache.py`  (+6/-13)
- `op_tests/triton_tests/attention/test_pa_decode.py`  (+8/-10)
- `op_tests/triton_tests/attention/test_chunked_pa_prefill.py`  (+6/-9)
- `op_tests/triton_tests/attention/test_pa_prefill.py`  (+5/-8)
- `op_tests/triton_tests/attention/test_fav3_sage.py`  (+4/-8)
- `op_tests/triton_tests/attention/test_unified_attention_sparse_mla.py`  (+2/-9)
- `op_tests/triton_tests/attention/test_unified_attention.py`  (+3/-5)
- `op_tests/triton_tests/gemm/basic/test_gemm_a8wfp4.py`  (+3/-5)
- `op_tests/triton_tests/attention/test_la.py`  (+2/-2)
- `op_tests/triton_tests/attention/test_mla_decode_rope.py`  (+2/-2)
- `op_tests/triton_tests/gemm/basic/test_gemm_a16w8_blockscale.py`  (+2/-2)

## Key added lines (kernel files)

**`op_tests/triton_tests/attention/test_chunked_pa_prefill.py`**
```
NUM_QUERIES_PER_KV = [1, 8]
HEAD_SIZES = [128]
SLIDING_WINDOW = [0, 256, 1024]
KV_CACHE_DTYPES = ["auto", "fp8e4m3"]
```

**`op_tests/triton_tests/attention/test_fav3_sage.py`**
```
HEAD_SZ = 128
HEAD_SZ = 128
layout = "bhsd"
hadamard_rotate = True  # hadamard expected to be on
```

**`op_tests/triton_tests/attention/test_flash_attn_kvcache.py`**
```
@pytest.mark.parametrize("mha_type", ["mha", "gqa"])
@pytest.mark.parametrize("d", [64, 128])
dtype = torch.bfloat16
def test_flash_attn_kvcache_hipgraph_capture(mha_type, new_kv):
```

**`op_tests/triton_tests/attention/test_la.py`**
```
sum(int(n) for n in n_ctx)
sum(int(n) for n in n_ctx)
```

**`op_tests/triton_tests/attention/test_mha.py`**
```
atol_floor = 5e-1 if is_forward else 1.0
def _test_mha_impl(
if CAUSAL and (SEQLEN_Q * SEQLEN_K > 128 * 128):
pytest.skip(
```
