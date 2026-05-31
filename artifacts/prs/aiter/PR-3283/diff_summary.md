# Diff summary

- **files changed:** 12
- **lines:** +581 / -84
- **kernel-ish files:** 5

## Files (by churn)

- `op_tests/op_benchmarks/triton/bench_batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant.py`  (+198/-0)
- `aiter/ops/triton/configs/gemm/gfx950-BATCHED_GEMM-A16WFP4-N=128-K=512.json`  (+74/-0)
- `aiter/ops/triton/configs/gemm/gfx950-BATCHED_GEMM-A16WFP4-N=8192-K=8192.json`  (+74/-0)
- `aiter/ops/triton/configs/gemm/gfx950-BATCHED_GEMM-A16WFP4.json`  (+74/-0)
- `op_tests/op_benchmarks/triton/bench_batched_gemm_bf16.py`  (+53/-17)
- `aiter/ops/triton/configs/gemm/gfx950-BATCHED_GEMM-A8W8-A_PER_TOKEN_GROUP_PREQUANT_W_PER_BATCHED_TENSOR_QUANT-N=8192-K=8192.json`  (+62/-0)
- `aiter/ops/triton/configs/gemm/gfx950-BATCHED_GEMM-A8W8-A_PER_TOKEN_GROUP_PREQUANT_W_PER_BATCHED_TENSOR_QUANT-N=512-K=128.json`  (+15/-21)
- `op_tests/op_benchmarks/triton/bench_batched_gemm_a16wfp4.py`  (+11/-15)
- `aiter/ops/triton/configs/gemm/gfx950-BATCHED_GEMM-A8W8-A_PER_TOKEN_GROUP_PREQUANT_W_PER_BATCHED_TENSOR_QUANT-N=128-K=512.json`  (+9/-15)
- `aiter/ops/triton/configs/gemm/gfx950-BATCHED_GEMM-A8W8-A_PER_TOKEN_GROUP_PREQUANT_W_PER_BATCHED_TENSOR_QUANT.json`  (+9/-15)
- `aiter/ops/triton/_triton_kernels/gemm/batched/batched_gemm_a16wfp4.py`  (+1/-1)
- `aiter/ops/triton/_triton_kernels/gemm/batched/batched_gemm_bf16.py`  (+1/-0)

## Key added lines (kernel files)

**`aiter/ops/triton/_triton_kernels/gemm/batched/batched_gemm_a16wfp4.py`**
```
return get_gemm_config("BATCHED_GEMM-A16WFP4", M, N, 2 * K)
```

**`op_tests/op_benchmarks/triton/bench_batched_gemm_a16wfp4.py`**
```
from aiter.ops.triton.gemm.batched.batched_gemm_a16wfp4 import (
batched_gemm_a16wfp4,
mem_read = (
x.numel() * x.element_size()
```

**`op_tests/op_benchmarks/triton/bench_batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant.py`**
```
import math
import torch
import triton
from aiter.ops.triton.gemm.batched.batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant import (
```

**`op_tests/op_benchmarks/triton/bench_batched_gemm_bf16.py`**
```
generate_batched_gemm_a16w16_inputs as generate_batched_gemm_bf16_inputs,
x, w, bias, y = generate_batched_gemm_bf16_inputs(
dtype=c_dtype,
layout=layout,
```
