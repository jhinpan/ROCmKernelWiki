# Diff summary

- **files changed:** 28
- **lines:** +90 / -75
- **kernel-ish files:** 28

## Files (by churn)

- `vllm/v1/attention/backends/flashinfer.py`  (+9/-10)
- `vllm/v1/attention/ops/triton_reshape_and_cache_flash.py`  (+11/-8)
- `vllm/v1/attention/backends/rocm_aiter_fa.py`  (+8/-7)
- `vllm/v1/attention/backends/flash_attn.py`  (+5/-5)
- `vllm/model_executor/layers/attention/mla_attention.py`  (+5/-4)
- `vllm/v1/attention/backends/mla/flashmla.py`  (+6/-3)
- `vllm/v1/attention/backends/triton_attn.py`  (+5/-4)
- `vllm/v1/attention/backends/rocm_attn.py`  (+4/-3)
- `vllm/v1/attention/backends/mla/flashinfer_mla.py`  (+3/-3)
- `vllm/v1/attention/backends/mla/flashinfer_mla_sparse.py`  (+3/-2)
- `vllm/v1/attention/backends/rocm_aiter_unified_attn.py`  (+3/-2)
- `vllm/platforms/cpu.py`  (+2/-2)
- `vllm/utils/torch_utils.py`  (+4/-0)
- `vllm/v1/attention/backend.py`  (+0/-4)
- `vllm/v1/attention/backends/mla/flashattn_mla.py`  (+2/-2)

## Key added lines (kernel files)

**`vllm/config/cache.py`**
```
from vllm.utils.torch_utils import is_quantized_kv_cache
if is_quantized_kv_cache(cache_dtype):
```

**`vllm/model_executor/layers/attention/mla_attention.py`**
```
is_quantized_kv_cache,
and is_quantized_kv_cache(kv_cache_dtype)
and is_quantized_kv_cache(kv_cache_dtype)
fp8_attention = is_quantized_kv_cache(self.kv_cache_dtype)
```

**`vllm/model_executor/layers/quantization/kv_cache.py`**
```
from vllm.utils.torch_utils import is_quantized_kv_cache
```

**`vllm/model_executor/models/extract_hidden_states.py`**
```
from vllm.utils.torch_utils import is_quantized_kv_cache, kv_cache_dtype_str_to_dtype
```

**`vllm/platforms/cpu.py`**
```
from vllm.utils.torch_utils import is_quantized_kv_cache
if is_quantized_kv_cache(cache_config.cache_dtype):
```
