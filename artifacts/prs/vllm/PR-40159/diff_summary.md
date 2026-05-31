# Diff summary

- **files changed:** 28
- **lines:** +243 / -146
- **kernel-ish files:** 28

## Files (by churn)

- `vllm/model_executor/layers/mamba/gdn_linear_attn.py`  (+36/-30)
- `vllm/model_executor/layers/kda.py`  (+25/-18)
- `vllm/model_executor/layers/activation.py`  (+15/-12)
- `vllm/model_executor/layers/attention/mla_attention.py`  (+17/-10)
- `vllm/model_executor/layers/sparse_attn_indexer.py`  (+14/-12)
- `vllm/model_executor/layers/fused_moe/prepare_finalize/flashinfer_nvlink_two_sided.py`  (+16/-9)
- `vllm/model_executor/layers/fused_moe/prepare_finalize/flashinfer_nvlink_one_sided.py`  (+18/-6)
- `vllm/model_executor/layers/attention/attention.py`  (+15/-6)
- `vllm/model_executor/layers/mamba/mamba_mixer2.py`  (+17/-4)
- `vllm/model_executor/layers/attention/cross_attention.py`  (+9/-4)
- `vllm/model_executor/layers/fused_moe/oracle/mxfp4.py`  (+8/-5)
- `vllm/model_executor/layers/mamba/mamba_mixer.py`  (+9/-4)
- `vllm/model_executor/layers/fused_moe/all2all_utils.py`  (+7/-4)
- `vllm/model_executor/layers/mamba/linear_attn.py`  (+5/-4)
- `vllm/model_executor/layers/mamba/short_conv.py`  (+5/-4)

## Key added lines (kernel files)

**`vllm/model_executor/layers/activation.py`**
```
"gelu_pytorch_tanh": lambda: _get_gelu_pytorch_tanh(),
def _get_gelu_pytorch_tanh() -> nn.Module:
"""Get PyTorch GELU with tanh approximation, with ROCm fallback."""
if current_platform.is_rocm():
```

**`vllm/model_executor/layers/attention/attention.py`**
```
AttentionMetadata,
sliding_window: int | None
self.impl = impl_cls(  # type: ignore[assignment]  # impl_cls always returns an AttentionImpl subclass
def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec | None:
```

**`vllm/model_executor/layers/attention/chunked_local_attention.py`**
```
underlying_attn_backend: type[AttentionBackend],
```

**`vllm/model_executor/layers/attention/cross_attention.py`**
```
underlying_attn_backend: type[AttentionBackend],
assert new_metadata.encoder_seq_lens_cpu is not None
attn_metadata.slot_mapping = slot_mapping  # type: ignore[attr-defined]
self.do_kv_cache_update(  # type: ignore[attr-defined]
```

**`vllm/model_executor/layers/attention/encoder_only_attention.py`**
```
underlying_attn_backend: type[AttentionBackend],
def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec | None:
```
