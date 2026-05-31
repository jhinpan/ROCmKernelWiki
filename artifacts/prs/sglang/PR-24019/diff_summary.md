# Diff summary

- **files changed:** 26
- **lines:** +92 / -534
- **kernel-ish files:** 26

## Files (by churn)

- `python/sglang/srt/layers/attention/nsa/tilelang_kernel.py`  (+1/-395)
- `python/sglang/srt/layers/attention/compressed/indexer.py`  (+76/-78)
- `python/sglang/srt/layers/attention/compressed/metadata.py`  (+11/-12)
- `python/sglang/srt/model_executor/cuda_graph_runner.py`  (+1/-9)
- `python/sglang/srt/layers/attention/debug_flash_mla_adapter.py`  (+0/-7)
- `python/sglang/srt/layers/attention/deepseek_v4_backend.py`  (+2/-4)
- `python/sglang/srt/layers/attention/hybrid_linear_attn_backend.py`  (+0/-4)
- `python/sglang/srt/models/deepseek_v2.py`  (+0/-4)
- `python/sglang/srt/layers/attention/base_attn_backend.py`  (+0/-2)
- `python/sglang/srt/layers/attention/hybrid_attn_backend.py`  (+0/-2)
- `python/sglang/srt/layers/attention/tbo_backend.py`  (+0/-2)
- `python/sglang/srt/models/deepseek_v4.py`  (+1/-1)
- `python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py`  (+0/-1)
- `python/sglang/srt/layers/attention/aiter_backend.py`  (+0/-1)
- `python/sglang/srt/layers/attention/cutlass_mla_backend.py`  (+0/-1)

## Key added lines (kernel files)

**`python/sglang/srt/layers/attention/compressed/indexer.py`**
```
from typing import TYPE_CHECKING, Any, List, Optional, Tuple
logits = page_table.new_empty((batch_size, max_seq_len), dtype=torch.float32)
for i in range(batch_size):
q = q_fp8[i, 0]  # (num_heads, head_dim)
```

**`python/sglang/srt/layers/attention/compressed/metadata.py`**
```
copy_fields = ["page_table", "c4_seq_lens"]
copy_fields = ["page_table", "c4_seq_lens", "deep_gemm_metadata"]
copy_metadata(
src=other,
```

**`python/sglang/srt/layers/attention/deepseek_v4_backend.py`**
```
out_cache_loc=torch.zeros_like(seq_lens),
```

**`python/sglang/srt/layers/attention/nsa/tilelang_kernel.py`**
```
from typing import Optional, Tuple
```

**`python/sglang/srt/model_executor/cuda_graph_runner.py`**
```
if self.buffers.out_cache_loc_swa is not None:
```
