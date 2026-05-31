# Diff summary

- **files changed:** 11
- **lines:** +279 / -9
- **kernel-ish files:** 6

## Files (by churn)

- `aiter/ops/triton/utils/moe_config_utils.py`  (+82/-0)
- `aiter/ops/triton/moe_configs/device_name=AMD_Instinct_MI300X,dtype=fp8_w8a8.json`  (+35/-0)
- `aiter/ops/triton/moe_configs/device_name=AMD_Instinct_MI300X,dtype=int4_w4a16.json`  (+35/-0)
- `aiter/ops/triton/moe_configs/device_name=AMD_Instinct_MI300X,dtype=int8_w8a16.json`  (+35/-0)
- `aiter/ops/triton/moe_configs/device_name=AMD_Instinct_MI300X,dtype=int8_w8a8.json`  (+35/-0)
- `aiter/ops/triton/moe_configs/device_name=AMD_Instinct_MI300X.json`  (+35/-0)
- `aiter/ops/triton/mha.py`  (+12/-4)
- `op_tests/triton/test_moe.py`  (+8/-2)
- `op_benchmarks/triton/bench_moe.py`  (+2/-2)
- `aiter/ops/triton/moe_op.py`  (+0/-1)
- `aiter/ops/triton/utils/__init__.py`  (+0/-0)

## Key added lines (kernel files)

**`aiter/ops/triton/mha.py`**
```
config = {
'BLOCK_M': 128,
'BLOCK_N': 32, # BLOCK_N: 64 spills for _attn_fwd
'waves_per_eu': 2,
```

**`aiter/ops/triton/utils/moe_config_utils.py`**
```
import torch
from typing import Any, Dict, Optional
import os
import json
```

**`op_benchmarks/triton/bench_moe.py`**
```
a, b, triton_out, _, b_zp, b_scale, topk_weights, topk_ids, sorted_token_ids, expert_ids, num_tokens_post_padded, config
a, b, triton_out, _, b_zp, a_scale, b_scale, topk_weights, topk_ids, sorted_token_ids, expert_ids, num_tokens_post_padde
```

**`op_tests/triton/test_moe.py`**
```
from aiter.ops.triton.utils.moe_config_utils import get_optimal_moe_config_func
moe_config_func = get_optimal_moe_config_func(dtype, use_int8_w8a16=int8_w8a16, use_fp8_w8a8=fp8_w8a8)
config = moe_config_func(M)
moe_config_func = get_optimal_moe_config_func(dtype, use_int4_w4a16=True)
```
