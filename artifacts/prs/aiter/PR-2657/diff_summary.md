# Diff summary

- **files changed:** 36
- **lines:** +2100 / -10
- **kernel-ish files:** 8

## Files (by churn)

- `aiter/ops/triton/attention/mla_decode.py`  (+761/-0)
- `aiter/ops/triton/configs/gemm/gfx1250-GEMM-A16WFP4.json`  (+86/-0)
- `aiter/ops/triton/configs/gemm/gfx1250-GEMM-AFP4WFP4_PRESHUFFLED.json`  (+86/-0)
- `aiter/ops/mha.py`  (+81/-1)
- `aiter/ops/triton/configs/gfx1250-MHA-DEFAULT.json`  (+81/-0)
- `aiter/ops/triton/configs/gemm/gfx1250-BATCHED_GEMM-AFP4WFP4.json`  (+80/-0)
- `aiter/ops/triton/configs/gemm/gfx1250-BATCHED_GEMM_PREQUANT-AFP4WFP4.json`  (+80/-0)
- `aiter/ops/triton/configs/gemm/gfx1250-GEMM-A16W16.json`  (+80/-0)
- `aiter/ops/triton/configs/gemm/gfx1250-GEMM-A8WFP4.json`  (+80/-0)
- `aiter/ops/triton/configs/gemm/gfx1250-GEMM-A16W16-gated.json`  (+74/-0)
- `aiter/ops/triton/configs/gemm/gfx1250-GEMM-AFP4WFP4.json`  (+74/-0)
- `aiter/ops/triton/configs/gemm/gfx1250-GEMM_PREQUANT-AFP4WFP4.json`  (+74/-0)
- `aiter/ops/triton/configs/gemm/gfx1250-BATCHED_GEMM-A8W8-A_PER_TOKEN_GROUP_PREQUANT_W_PER_BATCHED_TENSOR_QUANT.json`  (+68/-0)
- `aiter/ops/triton/configs/gfx1250-GMM.json`  (+51/-0)
- `aiter/ops/triton/configs/gemm/gfx1250-FF-A16W16-fused.json`  (+50/-0)

## Key added lines (kernel files)

**`aiter/jit/core.py`**
```
ENABLE_CK = int(os.environ.get("ENABLE_CK", "1")) != 0
if get_gfx() != "gfx942" and int(os.getenv("AITER_FP4x2", "1")) > 0:
```

**`aiter/ops/mha.py`**
```
from ..jit.core import CK_DIR, AITER_META_DIR, ENABLE_CK, compile_ops
if not ENABLE_CK:
from .triton.attention.mha import flash_attn_func as flash_attn_func_triton
return flash_attn_func_triton(
```

**`aiter/ops/triton/__init__.py`**
```
"mla_decode": "attention.mla_decode",
```

**`aiter/ops/triton/attention/mha.py`**
```
max_seqlen_q = int(max_seqlen_q)
max_seqlen_k = int(max_seqlen_k)
batch, seqlen_q, num_q_heads = (int(x) for x in q.shape[:-1])
```

**`aiter/ops/triton/attention/mla_decode.py`**
```
Triton MLA decode attention with split-K parallelization and paged KV cache.
Supports both MHA (kv_group_num==1) and GQA/MQA/MLA (kv_group_num>1).
import torch
import triton
```
