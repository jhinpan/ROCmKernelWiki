# Diff summary

- **files changed:** 20
- **lines:** +357 / -810
- **kernel-ish files:** 19

## Files (by churn)

- `python/triton_kernels/triton_kernels/tensor_details/layout_details/hopper_value.py`  (+21/-194)
- `python/triton_kernels/bench/bench_mlp.py`  (+83/-128)
- `python/triton_kernels/triton_kernels/matmul_ogs_details/_matmul_ogs.py`  (+44/-98)
- `python/triton_kernels/tests/test_matmul.py`  (+39/-93)
- `python/triton_kernels/triton_kernels/matmul_ogs.py`  (+19/-65)
- `python/triton_kernels/tests/test_tensor.py`  (+60/-1)
- `python/triton_kernels/triton_kernels/matmul_ogs_details/_finalize_matmul.py`  (+11/-46)
- `python/triton_kernels/triton_kernels/numerics_details/mxfp.py`  (+18/-24)
- `python/triton_kernels/triton_kernels/tensor.py`  (+9/-24)
- `python/triton_kernels/triton_kernels/matmul_ogs_details/opt_flags.py`  (+17/-15)
- `python/triton_kernels/triton_kernels/matmul_ogs_details/opt_flags_details/opt_flags_nvidia.py`  (+11/-17)
- `python/triton_kernels/triton_kernels/tensor_details/layout_details/hopper_scale.py`  (+6/-17)
- `python/triton_kernels/triton_kernels/numerics_details/mxfp_details/_downcast_to_mxfp.py`  (+6/-15)
- `python/triton_kernels/triton_kernels/matmul_ogs_details/_p_matmul_ogs.py`  (+5/-15)
- `python/triton_kernels/triton_kernels/tensor_details/layout.py`  (+0/-19)

## Key added lines (kernel files)

**`python/triton_kernels/bench/bench_mlp.py`**
```
from triton_kernels.numerics_details.mxfp import downcast_to_mxfp
from triton_kernels.numerics import InFlexData
from triton_kernels.routing import routing
from triton_kernels.tensor import convert_layout
```

**`python/triton_kernels/tests/test_matmul.py`**
```
from dataclasses import dataclass, fields
from triton_kernels.matmul_ogs import FlexCtx, PrecisionConfig, FusedActivation, FnSpecs
from triton_kernels.numerics_details.mxfp import downcast_to_mxfp, upcast_from_mxfp
def init_precision(out_dtype, weight_dtype, is_mixed_input, n_expts_tot=1, device="cuda"):
```

**`python/triton_kernels/tests/test_tensor.py`**
```
import torch
import pytest
import math
from triton_kernels.testing import assert_equal
```

**`python/triton_kernels/triton_kernels/compaction_details/_masked_compaction.py`**
```
rev_arange = tl.where(active_bits, 0, K - 1 - tl.arange(0, K))
yv = tl.where(active_bits, yv, sentinel)
yi = tl.where(active_bits, yi, sentinel)
```

**`python/triton_kernels/triton_kernels/matmul_ogs.py`**
```
from .tensor_details import layout
inp_flex.expected_scale,
if postprocessing_features.finalize and (opt_flags.split_k > 1 or not opt_flags.fused_scatter):
dtype = torch.float32 if opt_flags.split_k > 1 else out_dtype
```
