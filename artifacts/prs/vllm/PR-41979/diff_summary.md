# Diff summary

- **files changed:** 34
- **lines:** +586 / -552
- **kernel-ish files:** 32

## Files (by churn)

- `vllm/model_executor/layers/fused_moe/experts/triton_moe.py`  (+522/-0)
- `vllm/model_executor/layers/fused_moe/fused_moe.py`  (+0/-494)
- `vllm/model_executor/layers/fused_moe/__init__.py`  (+7/-5)
- `vllm/model_executor/layers/fused_moe/oracle/fp8.py`  (+4/-4)
- `vllm/model_executor/layers/fused_moe/oracle/mxfp4.py`  (+4/-4)
- `vllm/model_executor/layers/fused_moe/oracle/unquantized.py`  (+5/-3)
- `vllm/model_executor/layers/quantization/quark/quark_moe.py`  (+4/-4)
- `docs/design/moe_kernel_features.md`  (+3/-3)
- `tests/kernels/moe/test_flashinfer.py`  (+3/-3)
- `tests/kernels/moe/test_triton_moe_no_act_mul.py`  (+3/-3)
- `vllm/model_executor/layers/fused_moe/layer.py`  (+3/-3)
- `vllm/model_executor/layers/fused_moe/oracle/int_wna16.py`  (+3/-3)
- `tests/kernels/moe/modular_kernel_tools/mk_objects.py`  (+2/-2)
- `tests/kernels/moe/utils.py`  (+3/-1)
- `vllm/model_executor/layers/fused_moe/oracle/nvfp4.py`  (+2/-2)

## Key added lines (kernel files)

**`csrc/quantization/gguf/moe.cuh`**
```
based on ./vllm/model_executor/layers/fused_moe/experts/triton_moe.py */
```

**`tests/kernels/moe/modular_kernel_tools/mk_objects.py`**
```
from vllm.model_executor.layers.fused_moe.experts.flashinfer_cutlass_moe import (
from vllm.model_executor.layers.fused_moe.experts.rocm_aiter_moe import (
```

**`tests/kernels/moe/test_flashinfer.py`**
```
from vllm.model_executor.layers.fused_moe.experts.flashinfer_cutlass_moe import (
FlashInferExperts,
```

**`tests/kernels/moe/test_flashinfer_moe.py`**
```
from vllm.model_executor.layers.fused_moe.experts.flashinfer_cutlass_moe import (
```

**`tests/kernels/moe/test_marlin_vs_trtllm_mxint4.py`**
```
from vllm.model_executor.layers.fused_moe.experts.marlin_moe import (
```
