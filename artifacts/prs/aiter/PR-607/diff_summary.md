# Diff summary

- **files changed:** 28
- **lines:** +1990 / -133
- **kernel-ish files:** 18

## Files (by churn)

- `aiter/ops/triton/fused_qk_concat.py`  (+426/-0)
- `aiter/ops/triton/fused_mxfp4_quant.py`  (+300/-0)
- `aiter/ops/triton/gemm_a16w16_atomic.py`  (+237/-0)
- `op_tests/triton_tests/test_fused_mxfp4_quant.py`  (+134/-0)
- `aiter/ops/triton/fused_mul_add.py`  (+131/-0)
- `aiter/ops/triton/batched_gemm_afp4wfp4.py`  (+77/-50)
- `op_tests/triton_tests/test_fused_qk_concat.py`  (+111/-0)
- `aiter/ops/triton/configs/gemm/MI350X-GEMM-A16W16-ATOMIC-N=256-K=7168.json`  (+80/-0)
- `aiter/ops/triton/configs/gemm/MI350X-BATCHED_GEMM_PREQUANT-AFP4WFP4-N=128-K=512.json`  (+75/-0)
- `aiter/ops/triton/configs/gemm/MI350X-GEMM_PREQUANT-AFP4WFP4-N=512-K=7168.json`  (+75/-0)
- `aiter/ops/triton/configs/gemm/MI350X-BATCHED_GEMM_PREQUANT-AFP4WFP4-N=512-K=128.json`  (+74/-0)
- `aiter/ops/triton/configs/gemm/MI350X-GEMM_PREQUANT-AFP4WFP4.json`  (+37/-37)
- `op_tests/triton_tests/test_fused_mul_add.py`  (+61/-0)
- `aiter/ops/triton/gemm_afp4wfp4_pre_quant_atomic.py`  (+35/-8)
- `aiter/ops/triton/rope.py`  (+29/-0)

## Key added lines (kernel files)

**`aiter/ops/triton/batched_gemm_afp4wfp4.py`**
```
stride_in_ab,
stride_in_am,
stride_in_ak,
stride_in_bb,
```

**`aiter/ops/triton/batched_gemm_afp4wfp4_pre_quant.py`**
```
W is an e2m1 fp4 tensor and w_scales is an e8m0 tensor.
X gets quantized to the microscale fp4 (mxfp4) format before the GEMM.
if config["BLOCK_SIZE_K"] >= 2 * K:
config["BLOCK_SIZE_K"] = triton.next_power_of_2(2 * K)
```

**`aiter/ops/triton/fused_mul_add.py`**
```
import torch
import triton
import triton.language as tl
from typing import Optional
```

**`aiter/ops/triton/fused_mxfp4_quant.py`**
```
import torch
import triton
import triton.language as tl
from aiter.ops.triton.quant import _mxfp4_quant_op
```

**`aiter/ops/triton/fused_qk_concat.py`**
```
import torch
import triton
import triton.language as tl
from aiter.ops.triton.rope import _get_gptj_rotated_x_1D, _get_neox_rotated_x_1D
```
