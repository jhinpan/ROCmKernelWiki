# Diff summary

- **files changed:** 77
- **lines:** +1227 / -905
- **kernel-ish files:** 77

## Files (by churn)

- `python/sglang/srt/model_executor/model_runner.py`  (+107/-84)
- `python/sglang/srt/model_executor/cuda_graph_runner.py`  (+70/-67)
- `python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py`  (+64/-59)
- `python/sglang/srt/model_executor/piecewise_cuda_graph_runner.py`  (+60/-58)
- `python/sglang/srt/layers/attention/dsa/dsa_indexer.py`  (+43/-44)
- `python/sglang/srt/model_executor/forward_context.py`  (+84/-0)
- `python/sglang/srt/model_executor/cpu_graph_runner.py`  (+39/-38)
- `python/sglang/srt/batch_overlap/operations.py`  (+62/-7)
- `python/sglang/srt/layers/attention/dsv4/compress_hip.py`  (+37/-20)
- `python/sglang/srt/layers/attention/dsa_backend.py`  (+31/-24)
- `python/sglang/srt/layers/attention/flashattention_backend.py`  (+23/-27)
- `python/sglang/srt/layers/attention/xpu_backend.py`  (+23/-25)
- `python/sglang/srt/layers/attention/aiter_backend.py`  (+23/-24)
- `python/sglang/srt/speculative/frozen_kv_mtp_utils.py`  (+37/-10)
- `python/sglang/srt/speculative/multi_layer_eagle_draft_extend_cuda_graph_runner.py`  (+22/-25)

## Key added lines (kernel files)

**`python/sglang/srt/batch_overlap/operations.py`**
```
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, replace
from typing import (
TYPE_CHECKING,
```

**`python/sglang/srt/batch_overlap/two_batch_overlap.py`**
```
from sglang.srt.model_executor.forward_context import get_attn_backend
attn_backend = get_attn_backend()
assert isinstance(attn_backend, TboAttnBackend)
```

**`python/sglang/srt/hardware_backend/musa/attention/flashattention_backend.py`**
```
self.token_to_kv_pool.set_kv_buffer(
self.token_to_kv_pool.set_mla_kv_buffer(
key_cache, value_cache = self.token_to_kv_pool.get_kv_buffer(layer.layer_id)
kv_cache = self.token_to_kv_pool.get_key_buffer(layer.layer_id).to(
```

**`python/sglang/srt/hardware_backend/npu/attention/ascend_backend.py`**
```
def _cp_allgather_and_save_kv_npu(
forward_batch, layer, k, v, cp_size, token_to_kv_pool
token_to_kv_pool.set_kv_buffer(
self.req_to_token_pool = model_runner.req_to_token_pool
```

**`python/sglang/srt/hardware_backend/npu/attention/mla_preprocess.py`**
```
from sglang.srt.model_executor.forward_context import (
get_attn_backend,
get_token_to_kv_pool,
k_cache, v_cache = get_token_to_kv_pool().get_kv_buffer(self.layer_id)
```
