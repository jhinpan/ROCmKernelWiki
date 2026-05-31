# Diff summary

- **files changed:** 23
- **lines:** +2185 / -55
- **kernel-ish files:** 10

## Files (by churn)

- `aiter/ops/triton/batched_gemm_afp4wfp4_pre_quant.py`  (+433/-0)
- `aiter/ops/triton/gemm_afp4wfp4_pre_quant_atomic.py`  (+290/-0)
- `op_tests/op_benchmarks/triton/bench_batched_gemm_afp4wfp4_pre_quant.py`  (+177/-0)
- `op_tests/op_benchmarks/triton/bench_batched_gemm_afp4wfp4.py`  (+173/-0)
- `op_tests/triton_tests/test_batched_gemm_afp4wfp4_pre_quant.py`  (+156/-0)
- `op_tests/triton_tests/test_gemm_afp4wfp4_pre_quant_atomic.py`  (+144/-0)
- `aiter/ops/triton/batched_gemm_afp4wfp4.py`  (+49/-35)
- `aiter/ops/triton/configs/gemm/MI350X-BATCHED_GEMM_PREQUANT-AFP4WFP4.json`  (+82/-0)
- `aiter/ops/triton/configs/gemm/MI350X-BATCHED_GEMM-AFP4WFP4-N=128-K=512.json`  (+75/-0)
- `aiter/ops/triton/configs/gemm/MI350X-BATCHED_GEMM-AFP4WFP4-N=512-K=128.json`  (+75/-0)
- `aiter/ops/triton/configs/gemm/MI350X-GEMM-AFP4WFP4-N=2112-K=7168.json`  (+75/-0)
- `aiter/ops/triton/configs/gemm/MI350X-GEMM-AFP4WFP4-N=3072-K=1536.json`  (+75/-0)
- `aiter/ops/triton/configs/gemm/MI350X-GEMM-AFP4WFP4-N=512-K=7168.json`  (+75/-0)
- `aiter/ops/triton/configs/gemm/MI350X-GEMM-AFP4WFP4-N=7168-K=2048.json`  (+75/-0)
- `aiter/ops/triton/configs/gemm/MI350X-GEMM-AFP4WFP4-N=7168-K=256.json`  (+75/-0)

## Key added lines (kernel files)

**`aiter/ops/triton/batched_gemm_afp4wfp4.py`**
```
tl.assume(stride_ab > 0)
tl.assume(stride_bb > 0)
tl.assume(stride_cb > 0)
tl.assume(stride_asb > 0)
```

**`aiter/ops/triton/batched_gemm_afp4wfp4_pre_quant.py`**
```
from typing import Optional
import functools
import json
import os
```

**`aiter/ops/triton/gemm_a16w16.py`**
```
import os
cache_modifier: tl.constexpr,
b = tl.load(b_ptrs, cache_modifier=cache_modifier)
b = tl.load(
```

**`aiter/ops/triton/gemm_afp4wfp4.py`**
```
global _USE_GEMM_SPLITK_BF16
_USE_GEMM_SPLITK_BF16 = False
def set_use_gemm_splitk_bf16(value: bool):
global _USE_GEMM_SPLITK_BF16
```

**`aiter/ops/triton/gemm_afp4wfp4_pre_quant_atomic.py`**
```
from typing import Optional
import functools
import json
import os
```
