# Diff summary

- **files changed:** 17
- **lines:** +3678 / -70
- **kernel-ish files:** 17

## Files (by churn)

- `python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py`  (+1265/-0)
- `python/sglang/srt/layers/attention/nsa/tilelang_kernel.py`  (+1214/-2)
- `python/sglang/srt/layers/attention/dsv4/compress_hip.py`  (+455/-0)
- `python/sglang/srt/layers/attention/hip_flash_mla.py`  (+197/-0)
- `python/sglang/srt/layers/deepseek_v4_rope.py`  (+168/-0)
- `python/sglang/srt/layers/quantization/fp8.py`  (+143/-16)
- `python/sglang/srt/mem_cache/deepseek_v4_compress_state.py`  (+88/-21)
- `python/sglang/srt/models/deepseek_v4.py`  (+53/-5)
- `python/sglang/jit_kernel/deepseek_v4.py`  (+26/-0)
- `python/sglang/srt/layers/attention/attention_registry.py`  (+17/-4)
- `python/sglang/srt/layers/attention/dsv4/indexer.py`  (+8/-11)
- `python/sglang/srt/layers/attention/dsv4/compressor.py`  (+16/-2)
- `python/sglang/srt/mem_cache/deepseek_v4_memory_pool.py`  (+13/-4)
- `python/sglang/srt/environ.py`  (+7/-0)
- `python/sglang/srt/models/deepseek_v2.py`  (+5/-0)

## Key added lines (kernel files)

**`python/sglang/jit_kernel/deepseek_v4.py`**
```
from sglang.srt.utils import get_bool_env_var, is_hip
_is_hip = is_hip()
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip
if _use_aiter:
```

**`python/sglang/srt/environ.py`**
```
SGLANG_OPT_DPSK_V4_RADIX = EnvBool(True)
SGLANG_OPT_USE_OLD_COMPRESSOR = EnvBool(False)
SGLANG_OPT_USE_TRITON_SWA_PREPARE = EnvBool(True)
SGLANG_OPT_USE_AITER_MHC_PRE = EnvBool(True)
```

**`python/sglang/srt/layers/attention/attention_registry.py`**
```
from sglang.srt.utils import is_hip
if is_hip():
from sglang.srt.layers.attention.deepseek_v4_backend_hip_radix import (
DeepseekV4HipRadixBackend,
```

**`python/sglang/srt/layers/attention/deepseek_v4_backend_hip_radix.py`**
```
from __future__ import annotations
import enum
import functools
import logging
```

**`python/sglang/srt/layers/attention/dsv4/compress_hip.py`**
```
from __future__ import annotations
import os
from functools import cached_property
from typing import TYPE_CHECKING, Any
```
