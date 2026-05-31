# Diff summary

- **files changed:** 252
- **lines:** +622 / -447
- **kernel-ish files:** 252

## Files (by churn)

- `aiter/ops/triton/__init__.py`  (+136/-0)
- `aiter/ops/triton/quant/__init__.py`  (+44/-0)
- `aiter/ops/triton/attention/pa_mqa_logits.py`  (+8/-9)
- `aiter/ops/triton/rope/rope.py`  (+12/-5)
- `op_tests/triton_tests/fusions/test_fused_kv_cache.py`  (+8/-9)
- `aiter/ops/triton/quant/quant.py`  (+11/-3)
- `op_tests/op_benchmarks/triton/bench_ff_a16w16_fused.py`  (+10/-3)
- `aiter/ops/triton/_triton_kernels/gemm/basic/gemm_a16w16.py`  (+7/-4)
- `aiter/ops/triton/comms/fused/reduce_scatter_rmsnorm_quant_all_gather.py`  (+5/-6)
- `aiter/ops/triton/_triton_kernels/attention/extend_attention.py`  (+5/-5)
- `aiter/ops/triton/_triton_kernels/attention/mha.py`  (+5/-5)
- `aiter/ops/triton/_triton_kernels/attention/mha_fused_bwd.py`  (+5/-5)
- `aiter/ops/triton/_triton_kernels/attention/mla_decode_rope.py`  (+5/-5)
- `aiter/ops/triton/_triton_kernels/gemm/basic/gemm_a16wfp4.py`  (+5/-5)
- `aiter/ops/triton/_triton_kernels/gemm/batched/batched_gemm_a16wfp4.py`  (+5/-5)

## Key added lines (kernel files)

**`aiter/fused_moe.py`**
```
from aiter.ops.triton.quant.fused_mxfp4_quant import fused_dynamic_mxfp4_quant_moe_sort
```

**`aiter/ops/triton/__init__.py`**
```
import importlib.util
import sys
from types import SimpleNamespace
These following help implement backward-compatibility
```

**`aiter/ops/triton/_triton_kernels/activation.py`**
```
from .quant.quant import _mxfp4_quant_op
from .quant.fused_fp8_quant import _fp8_quant_op
```

**`aiter/ops/triton/_triton_kernels/attention/chunked_pa_prefill.py`**
```
from aiter.ops.triton.utils._triton.kernel_repr import make_kernel_repr
```

**`aiter/ops/triton/_triton_kernels/attention/extend_attention.py`**
```
from aiter.ops.triton._triton_kernels.activation import _tanh
from aiter.ops.triton.utils._triton.pid_preprocessing import remap_xcd
from aiter.ops.triton.utils._triton import arch_info
from aiter.ops.triton.utils.core import AITER_TRITON_CONFIGS_PATH
```
