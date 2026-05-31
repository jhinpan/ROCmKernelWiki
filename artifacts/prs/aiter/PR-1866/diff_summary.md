# Diff summary

- **files changed:** 73
- **lines:** +53 / -124
- **kernel-ish files:** 73

## Files (by churn)

- `aiter/ops/mha.py`  (+9/-10)
- `op_tests/triton_tests/normalization/test_rmsnorm.py`  (+8/-10)
- `op_tests/test_mha.py`  (+5/-5)
- `op_tests/test_mha_varlen.py`  (+5/-5)
- `aiter/rotary_embedding.py`  (+4/-5)
- `aiter/ops/triton/attention/pa_prefill.py`  (+2/-4)
- `aiter/test_common.py`  (+2/-4)
- `csrc/ck_gemm_a8w8_bpreshuffle/gemm_a8w8_bpreshuffle_tune.py`  (+3/-3)
- `csrc/cpp_itfs/utils.py`  (+2/-4)
- `aiter/ops/triton/_triton_kernels/attention/pod_attention.py`  (+2/-3)
- `aiter/ops/gemm_op_a8w8.py`  (+1/-3)
- `csrc/ck_gemm_a8w8_blockscale/gemm_a8w8_blockscale_tune.py`  (+2/-2)
- `csrc/cpp_itfs/pa/pa_ragged_test.py`  (+1/-3)
- `op_tests/test_pa_ragged.py`  (+1/-3)
- `op_tests/test_pa_v1.py`  (+1/-3)

## Key added lines (kernel files)

**`aiter/ops/gemm_op_a8w8.py`**
```
assert bias is not None, "Use asm gemm must give bias, please give a \
```

**`aiter/ops/mha.py`**
```
_, seqlen_q, _, _ = q.shape
_, seqlen_q, nhead_q, hdim_q = q.shape
_, seqlen_k, nhead_k, hdim_v = v.shape
_, seqlen_q, nhead_q, hdim_q = q.shape
```

**`aiter/ops/triton/_triton_kernels/attention/pod_attention.py`**
```
cu_id, se_id, xcc_id = tl.inline_asm_elementwise(
cu_id, se_id, xcc_id = get_cu_id()
```

**`aiter/ops/triton/attention/pa_prefill.py`**
```
raise ValueError("kv_cache_dtype='auto' unsupported for\
FP8 KV Cache prefill kernel")
```

**`aiter/rotary_embedding.py`**
```
raise ValueError(f"`Phi3LongRoPEScaledRotaryEmbedding` does not support \
rotary_dim != head_size ({rotary_dim}!={head_size}).")
q_cache, qc_cache, k_cache, qc_no_clamp_cache, q_inter_cache = (
```
