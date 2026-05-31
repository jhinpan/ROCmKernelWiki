# Diff summary

- **files changed:** 26
- **lines:** +534 / -92
- **kernel-ish files:** 26

## Files (by churn)

- `python/sglang/srt/layers/attention/nsa/tilelang_kernel.py`  (+395/-1)
- `python/sglang/srt/layers/attention/compressed/indexer.py`  (+78/-76)
- `python/sglang/srt/layers/attention/compressed/metadata.py`  (+12/-11)
- `python/sglang/srt/model_executor/cuda_graph_runner.py`  (+9/-1)
- `python/sglang/srt/layers/attention/debug_flash_mla_adapter.py`  (+7/-0)
- `python/sglang/srt/layers/attention/deepseek_v4_backend.py`  (+4/-2)
- `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`  (+4/-0)
- `python/sglang/srt/models/deepseek_v2.py`  (+4/-0)
- `python/sglang/srt/layers/attention/base_attn_backend.py`  (+2/-0)
- `python/sglang/srt/layers/attention/hybrid_attn_backend.py`  (+2/-0)
- `python/sglang/srt/layers/attention/tbo_backend.py`  (+2/-0)
- `python/sglang/srt/models/deepseek_v4.py`  (+1/-1)
- `python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py`  (+1/-0)
- `python/sglang/srt/layers/attention/aiter_backend.py`  (+1/-0)
- `python/sglang/srt/layers/attention/cutlass_mla_backend.py`  (+1/-0)

## Key added lines (kernel files)

**`python/sglang/srt/layers/attention/base_attn_backend.py`**
```
out_cache_loc: Optional[torch.Tensor] = None,
actual_forward_mode: Optional[ForwardMode] = None,
```

**`python/sglang/srt/layers/attention/compressed/indexer.py`**
```
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple
_arange_cache: Dict[str, torch.Tensor] = {}
"""Vectorized implementation that avoids .item() and Python loops,
making it compatible with CUDA graph capture."""
```

**`python/sglang/srt/layers/attention/compressed/metadata.py`**
```
copy_metadata(
src=other,
dst=self,
check_eq_fields=["page_size", "deep_gemm_metadata"],
```

**`python/sglang/srt/layers/attention/debug_flash_mla_adapter.py`**
```
from sglang.srt.layers.attention.nsa.tilelang_kernel import (
dpsk_v4_bf16_sparse_attention_fwd,
if backend == "tilelang":
return dpsk_v4_bf16_sparse_attention_fwd(**kwargs)
```

**`python/sglang/srt/layers/attention/deepseek_v4_backend.py`**
```
out_cache_loc=torch.zeros(
seq_lens.shape, dtype=torch.int64, device=seq_lens.device
```
