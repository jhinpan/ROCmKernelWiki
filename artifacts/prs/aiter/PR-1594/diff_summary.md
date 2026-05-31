# Diff summary

- **files changed:** 24
- **lines:** +7662 / -0
- **kernel-ish files:** 24

## Files (by churn)

- `aiter/ops/triton/_triton_kernels/gated_delta_rule/decode/causal_conv1d_split_qkv.py`  (+1101/-0)
- `aiter/ops/triton/_triton_kernels/gated_delta_rule/prefill/chunk_o.py`  (+796/-0)
- `aiter/ops/triton/_triton_kernels/gated_delta_rule/prefill/chunk_delta_h.py`  (+707/-0)
- `aiter/ops/triton/_triton_kernels/gated_delta_rule/utils/solve_tril.py`  (+589/-0)
- `aiter/ops/triton/_triton_kernels/gated_delta_rule/fused_qkvzba_split.py`  (+580/-0)
- `aiter/ops/triton/_triton_kernels/gated_delta_rule/gated_delta_rule_utils.py`  (+553/-0)
- `op_tests/triton_tests/test_gated_delta_rule.py`  (+537/-0)
- `aiter/ops/triton/_triton_kernels/gated_delta_rule/prefill/causal_conv1d_fwd_split_qkv.py`  (+399/-0)
- `aiter/ops/triton/_triton_kernels/gated_delta_rule/utils/l2norm.py`  (+344/-0)
- `aiter/ops/triton/gated_delta_net/gated_delta_rule.py`  (+335/-0)
- `aiter/ops/triton/_triton_kernels/gated_delta_rule/utils/cumsum.py`  (+323/-0)
- `aiter/ops/triton/_triton_kernels/gated_delta_rule/utils/wy_representation.py`  (+311/-0)
- `aiter/ops/triton/_triton_kernels/gated_delta_rule/decode/fused_sigmoid_gating_recurrent.py`  (+256/-0)
- `aiter/ops/triton/_triton_kernels/gated_delta_rule/decode/fused_recurrent.py`  (+191/-0)
- `aiter/ops/triton/_triton_kernels/gated_delta_rule/prefill/fused_cumsum_kkt.py`  (+135/-0)

## Key added lines (kernel files)

**`aiter/ops/triton/_triton_kernels/gated_delta_rule/__init__.py`**
```
Gated Delta Rule Operations (Forward Only).
This module provides optimized Triton kernels for gated delta rule computations.
Available operations:
- Fused recurrent forward: _fused_recurrent_gated_delta_rule_fwd_kernel
```

**`aiter/ops/triton/_triton_kernels/gated_delta_rule/decode/__init__.py`**
```
Gated Delta Rule Decode Operations (Forward Only).
This module provides optimized Triton kernels for decode/inference operations.
from .fused_recurrent import _fused_recurrent_gated_delta_rule_fwd_kernel
from .fused_sigmoid_gating_recurrent import fused_sigmoid_gating_delta_rule_update
```

**`aiter/ops/triton/_triton_kernels/gated_delta_rule/decode/causal_conv1d_split_qkv.py`**
```
"""Optimized causal_conv1d_update: directly output split q/k/v."""
import torch
import triton
import triton.experimental.gluon.language as gl
```

**`aiter/ops/triton/_triton_kernels/gated_delta_rule/decode/fused_recurrent.py`**
```
Fused recurrent gated delta rule forward kernel (Forward only).
This module provides an optimized fused recurrent implementation of the gated delta rule.
Note: Only forward pass is implemented. Backward pass is not supported in aiter.
import triton
```

**`aiter/ops/triton/_triton_kernels/gated_delta_rule/decode/fused_sigmoid_gating_recurrent.py`**
```
from typing import Optional
import torch
import triton
import triton.language as tl
```
