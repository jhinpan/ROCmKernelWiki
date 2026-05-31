# Diff summary

- **files changed:** 64
- **lines:** +5918 / -28
- **kernel-ish files:** 6

## Files (by churn)

- `op_tests/triton_tests/triton_metadata_redirect/triton_metadata_redirect/test_metadata_redirect.py`  (+5872/-0)
- `op_tests/triton_tests/triton_metadata_redirect/triton_metadata_redirect/kernel.py`  (+8/-0)
- `aiter/ops/triton/gemm_afp4wfp4.py`  (+4/-1)
- `op_tests/triton_tests/test_gemm_afp4wfp4.py`  (+2/-3)
- `aiter/ops/triton/fused_mul_add.py`  (+1/-2)
- `aiter/ops/triton/configs/gemm/MI350X-GEMM-AFP4WFP4_PRESHUFFLED-N=57344-K=8192.json`  (+1/-1)
- `aiter/ops/triton/configs/gemm/aot/README.md`  (+2/-0)
- `aiter/ops/triton/configs/gemm/aot/_gemm_afp4_wfp4_kernel_preshuffled_weight_scales_M=16-N=10240-K=8192/_gemm_afp4_wfp4_kernel_preshuffled_weight_scales.json`  (+1/-1)
- `aiter/ops/triton/configs/gemm/aot/_gemm_afp4_wfp4_kernel_preshuffled_weight_scales_M=16-N=57344-K=8192/_gemm_afp4_wfp4_kernel_preshuffled_weight_scales.json`  (+1/-1)
- `aiter/ops/triton/configs/gemm/aot/_gemm_afp4_wfp4_kernel_preshuffled_weight_scales_M=16-N=8192-K=28672/_gemm_afp4_wfp4_kernel_preshuffled_weight_scales.json`  (+1/-1)
- `aiter/ops/triton/configs/gemm/aot/_gemm_afp4_wfp4_kernel_preshuffled_weight_scales_M=16-N=8192-K=8192/_gemm_afp4_wfp4_kernel_preshuffled_weight_scales.json`  (+1/-1)
- `aiter/ops/triton/configs/gemm/aot/_gemm_afp4_wfp4_kernel_preshuffled_weight_scales_M=32-N=10240-K=8192/_gemm_afp4_wfp4_kernel_preshuffled_weight_scales.json`  (+1/-1)
- `aiter/ops/triton/configs/gemm/aot/_gemm_afp4_wfp4_kernel_preshuffled_weight_scales_M=32-N=57344-K=8192/_gemm_afp4_wfp4_kernel_preshuffled_weight_scales.json`  (+1/-1)
- `aiter/ops/triton/configs/gemm/aot/_gemm_afp4_wfp4_kernel_preshuffled_weight_scales_M=32-N=8192-K=28672/_gemm_afp4_wfp4_kernel_preshuffled_weight_scales.json`  (+1/-1)
- `aiter/ops/triton/configs/gemm/aot/_gemm_afp4_wfp4_kernel_preshuffled_weight_scales_M=32-N=8192-K=8192/_gemm_afp4_wfp4_kernel_preshuffled_weight_scales.json`  (+1/-1)

## Key added lines (kernel files)

**`aiter/ops/triton/fused_mul_add.py`**
```
f"FUSED_MUL_ADD: x={tuple(x.shape)} a={tuple(a.shape) if isinstance(a, torch.Tensor) else a} b={tuple(b.shape) if isinst
```

**`aiter/ops/triton/gemm_afp4wfp4.py`**
```
M_POW2 = triton.next_power_of_2(M)
if M < 32 and M_POW2 > 16:
M_POW2 = 16
metadata_pth = f"{AITER_TRITON_CONFIGS_PATH}/gemm/aot/{_gemm_afp4_wfp4_kernel_preshuffled_weight_scales.fn.__name__}_M={
```

**`op_tests/triton_tests/test_gemm_afp4wfp4.py`**
```
x_scales_shuffled = x_scales.contiguous()
[(False, False), (True, False), (True, True)],
```

**`op_tests/triton_tests/triton_metadata_redirect/triton_metadata_redirect/kernel.py`**
```
import triton
import triton.language as tl
@triton.jit
def empty_kernel(x_ptr, SIZE: tl.constexpr):
```

**`op_tests/triton_tests/triton_metadata_redirect/triton_metadata_redirect/test_metadata_redirect.py`**
```
import os
from pathlib import Path
import torch
import tempfile
```
