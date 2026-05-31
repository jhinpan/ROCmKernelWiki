# Diff summary

- **files changed:** 19
- **lines:** +55 / -22
- **kernel-ish files:** 18

## Files (by churn)

- `docs/design/attention_backends.md`  (+19/-19)
- `vllm/v1/attention/backends/flash_attn.py`  (+6/-1)
- `vllm/v1/attention/backend.py`  (+5/-1)
- `vllm/v1/attention/backends/flex_attention.py`  (+5/-1)
- `vllm/v1/attention/backends/tree_attn.py`  (+6/-0)
- `vllm/config/cache.py`  (+1/-0)
- `vllm/v1/attention/backends/flashinfer.py`  (+1/-0)
- `vllm/v1/attention/backends/mla/cutlass_mla.py`  (+1/-0)
- `vllm/v1/attention/backends/mla/flashattn_mla.py`  (+1/-0)
- `vllm/v1/attention/backends/mla/flashinfer_mla.py`  (+1/-0)
- `vllm/v1/attention/backends/mla/flashinfer_mla_sparse.py`  (+1/-0)
- `vllm/v1/attention/backends/mla/flashmla.py`  (+1/-0)
- `vllm/v1/attention/backends/mla/rocm_aiter_mla.py`  (+1/-0)
- `vllm/v1/attention/backends/mla/rocm_aiter_mla_sparse.py`  (+1/-0)
- `vllm/v1/attention/backends/mla/triton_mla.py`  (+1/-0)

## Key added lines (kernel files)

**`vllm/config/cache.py`**
```
"float16",
```

**`vllm/v1/attention/backend.py`**
```
supported_kv_cache_dtypes: ClassVar[list["CacheDType"]] = [
"float16",
"bfloat16",
```

**`vllm/v1/attention/backends/flash_attn.py`**
```
supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
"float16",
"bfloat16",
return kv_cache_dtype in ["auto", "float16", "bfloat16"]
```

**`vllm/v1/attention/backends/flashinfer.py`**
```
"float16",
```

**`vllm/v1/attention/backends/flex_attention.py`**
```
supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
"float16",
"bfloat16",
```
