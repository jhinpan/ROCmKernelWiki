# Diff summary

- **files changed:** 19
- **lines:** +1066 / -1575
- **kernel-ish files:** 19

## Files (by churn)

- `python/sglang/srt/layers/attention/triton_backend.py`  (+312/-362)
- `python/sglang/srt/layers/attention/aiter_backend.py`  (+15/-418)
- `python/sglang/srt/layers/attention/flashinfer_backend.py`  (+89/-150)
- `python/sglang/srt/layers/attention/flashattention_backend.py`  (+100/-96)
- `python/sglang/srt/layers/attention/flashmla_backend.py`  (+49/-135)
- `python/sglang/srt/layers/attention/wave_backend.py`  (+59/-71)
- `python/sglang/srt/layers/attention/trtllm_mha_backend.py`  (+58/-69)
- `python/sglang/srt/layers/attention/trtllm_mla_backend.py`  (+50/-53)
- `python/sglang/srt/layers/attention/flashinfer_mla_backend.py`  (+25/-49)
- `test/registered/attention/unittests/dense/test_tbo.py`  (+72/-2)
- `python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py`  (+30/-35)
- `python/sglang/srt/layers/attention/deepseek_v4_backend.py`  (+31/-34)
- `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py`  (+31/-34)
- `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`  (+35/-19)
- `python/sglang/srt/layers/attention/dsa_backend.py`  (+43/-5)

## Key added lines (kernel files)

**`python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py`**
```
def _init_cuda_graph_metadata(
seq_lens: torch.Tensor,
) -> "ForwardMetadata":
"""Create and store the per-bs ForwardMetadata for CUDA graph capture."""
```

**`python/sglang/srt/hardware_backend/npu/attention/ascend_gdn_backend.py`**
```
self.init_forward_metadata_replay_cuda_graph(
req_pool_indices=req_pool_indices,
seq_lens=seq_lens,
seq_lens_sum=None,
```

**`python/sglang/srt/layers/attention/aiter_backend.py`**
```
self.init_forward_metadata_replay_cuda_graph(
req_pool_indices=req_pool_indices,
seq_lens=seq_lens,
seq_lens_sum=None,
```

**`python/sglang/srt/layers/attention/cutlass_mla_backend.py`**
```
if forward_mode.is_decode_or_idle() and spec_info is None:
self.init_forward_metadata_replay_cuda_graph(
req_pool_indices=req_pool_indices,
seq_lens=seq_lens,
```

**`python/sglang/srt/layers/attention/deepseek_v4_backend.py`**
```
from types import SimpleNamespace
dummy_cache_loc = torch.zeros_like(seq_lens)
dummy_cache_loc = torch.zeros(num_tokens, **self.cuda_int32_kwargs)
dummy_cache_loc = None
```
