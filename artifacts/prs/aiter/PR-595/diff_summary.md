# Diff summary

- **files changed:** 16
- **lines:** +642 / -222
- **kernel-ish files:** 10

## Files (by churn)

- `aiter/ops/triton/mla_decode_rope.py`  (+69/-75)
- `aiter/ops/triton/moe_routing_sigmoid_top1_fused.py`  (+30/-62)
- `aiter/ops/triton/norm.py`  (+86/-0)
- `aiter/ops/triton/mha.py`  (+80/-4)
- `aiter/ops/triton/configs/moe/MI300X-MOE_ROUTING_SIGMOID_TOPK1.json`  (+70/-0)
- `aiter/ops/triton/extend_attention.py`  (+37/-29)
- `aiter/ops/triton/mha_onekernel_bwd.py`  (+33/-30)
- `aiter/ops/triton/configs/MI350X-MHA-DEFAULT.json`  (+61/-0)
- `aiter/ops/triton/configs/MI300X-MHA-DEFAULT.json`  (+60/-0)
- `aiter/ops/triton/mha_fused_bwd.py`  (+38/-20)
- `op_tests/triton_tests/test_moe_routing_sigmoid_top1_fused.py`  (+32/-2)
- `aiter/ops/triton/configs/MI300X-MLA_DECODE_ROPE-DEFAULT.json`  (+16/-0)
- `aiter/ops/triton/configs/MI350X-MLA_DECODE_ROPE-DEFAULT.json`  (+16/-0)
- `aiter/ops/triton/configs/MI300X-EXTEND_ATTENTION.json`  (+10/-0)
- `op_tests/triton_tests/test_mla_decode_rope.py`  (+4/-0)

## Key added lines (kernel files)

**`aiter/ops/triton/extend_attention.py`**
```
from typing import Optional
import functools
import json
from aiter.ops.triton.activation import _tanh
```

**`aiter/ops/triton/mha.py`**
```
from typing import Optional, Tuple
import functools
import json
from aiter.ops.triton.utils.core import AITER_TRITON_CONFIGS_PATH
```

**`aiter/ops/triton/mha_fused_bwd.py`**
```
from typing import Optional, Dict
import functools
import json
import aiter.ops.triton.utils.arch_info as arch_info
```

**`aiter/ops/triton/mha_onekernel_bwd.py`**
```
from typing import Optional, Dict
import functools
import json
import aiter.ops.triton.utils.arch_info as arch_info
```

**`aiter/ops/triton/mla_decode_rope.py`**
```
from typing import Optional
import functools
import json
from aiter.ops.triton.activation import _tanh
```
