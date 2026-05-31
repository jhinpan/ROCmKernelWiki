# Diff summary

- **files changed:** 22
- **lines:** +13 / -3
- **kernel-ish files:** 22

## Files (by churn)

- `op_tests/triton_tests/utils/__init__.py`  (+5/-0)
- `op_tests/triton_tests/test_mla_decode_rope.py`  (+2/-2)
- `op_tests/triton_tests/__init__.py`  (+3/-0)
- `op_tests/__init__.py`  (+2/-0)
- `op_tests/test_mla.py`  (+1/-1)
- `op_tests/triton_tests/test_batched_gemm_a8w8.py`  (+0/-0)
- `op_tests/triton_tests/test_batched_gemm_bf16.py`  (+0/-0)
- `op_tests/triton_tests/test_gemm_a16w16.py`  (+0/-0)
- `op_tests/triton_tests/test_gemm_a8w8.py`  (+0/-0)
- `op_tests/triton_tests/test_gemm_a8w8_blockscale.py`  (+0/-0)
- `op_tests/triton_tests/test_layernorm.py`  (+0/-0)
- `op_tests/triton_tests/test_mha.py`  (+0/-0)
- `op_tests/triton_tests/test_moe.py`  (+0/-0)
- `op_tests/triton_tests/test_moe_align_block_size.py`  (+0/-0)
- `op_tests/triton_tests/test_moe_e2e.py`  (+0/-0)

## Key added lines (kernel files)

**`op_tests/test_mla.py`**
```
from op_tests.triton_tests.utils import mla_decode_ref, mla_extend_ref
```

**`op_tests/triton_tests/__init__.py`**
```
from .utils import *
```

**`op_tests/triton_tests/test_mla_decode_rope.py`**
```
from op_tests.triton_tests.utils.mla_decode_ref import (
from op_tests.triton_tests.utils.rotary_embedding import DeepseekScalingRotaryEmbedding
```

**`op_tests/triton_tests/utils/__init__.py`**
```
from .mla_decode_ref import *
from .mla_extend_ref import *
from .rotary_embedding import *
```
