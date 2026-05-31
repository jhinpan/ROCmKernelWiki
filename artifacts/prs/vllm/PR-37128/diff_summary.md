# Diff summary

- **files changed:** 18
- **lines:** +1695 / -1369
- **kernel-ish files:** 17

## Files (by churn)

- `vllm/model_executor/layers/quantization/mxfp4.py`  (+180/-987)
- `vllm/model_executor/layers/fused_moe/oracle/mxfp4.py`  (+847/-0)
- `vllm/model_executor/layers/fused_moe/experts/trtllm_mxfp4_moe.py`  (+352/-0)
- `vllm/model_executor/layers/fused_moe/trtllm_moe.py`  (+0/-184)
- `vllm/model_executor/layers/fused_moe/gpt_oss_triton_kernels_moe.py`  (+151/-20)
- `vllm/model_executor/layers/quantization/utils/marlin_utils_fp4.py`  (+117/-3)
- `tests/kernels/quantization/test_mxfp4_triton_ep.py`  (+0/-83)
- `vllm/model_executor/layers/quantization/utils/mxfp4_utils.py`  (+1/-32)
- `vllm/model_executor/layers/fused_moe/layer.py`  (+0/-28)
- `vllm/model_executor/layers/fused_moe/rocm_aiter_fused_moe.py`  (+13/-6)
- `vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe.py`  (+11/-5)
- `vllm/model_executor/layers/fused_moe/oracle/nvfp4.py`  (+0/-11)
- `vllm/model_executor/layers/quantization/quark/quark_moe.py`  (+5/-5)
- `tests/kernels/moe/test_gpt_oss_triton_kernels.py`  (+8/-0)
- `tests/compile/fusions_e2e/conftest.py`  (+4/-1)

## Key added lines (kernel files)

**`tests/compile/fusions_e2e/conftest.py`**
```
from .common import is_blackwell
if is_blackwell():
monkeypatch.setenv("VLLM_USE_FLASHINFER_MOE_MXFP4_MXFP8", "1")
```

**`tests/kernels/moe/test_gpt_oss_triton_kernels.py`**
```
from vllm.platforms import current_platform
import triton_kernels.matmul_ogs_details.opt_flags as opt_flags
if current_platform.is_device_capability_family(100):
constraints = {
```

**`tests/kernels/moe/test_ocp_mx_moe.py`**
```
compilation_config={"cudagraph_capture_sizes": [16]},
```

**`vllm/model_executor/layers/fused_moe/experts/trtllm_mxfp4_moe.py`**
```
import torch
import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
```

**`vllm/model_executor/layers/fused_moe/fused_marlin_moe.py`**
```
kMxfp4Static,
kMxfp4Static,
```
