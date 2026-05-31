# Diff summary

- **files changed:** 13
- **lines:** +128 / -3
- **kernel-ish files:** 13

## Files (by churn)

- `aiter/ops/triton/utils/logger.py`  (+47/-0)
- `aiter/ops/triton/gemm_a8wfp4.py`  (+9/-2)
- `aiter/ops/triton/gemm_afp4wfp4_pre_quant_atomic.py`  (+9/-1)
- `aiter/ops/triton/gemm_afp4wfp4.py`  (+9/-0)
- `aiter/ops/triton/batched_gemm_afp4wfp4_pre_quant.py`  (+8/-0)
- `aiter/ops/triton/gemm_a16w16_atomic.py`  (+8/-0)
- `aiter/ops/triton/gemm_a8w8.py`  (+8/-0)
- `aiter/ops/triton/gemm_a8w8_blockscale.py`  (+7/-0)
- `aiter/ops/triton/batched_gemm_a8w8.py`  (+6/-0)
- `aiter/ops/triton/batched_gemm_afp4wfp4.py`  (+5/-0)
- `aiter/ops/triton/gemm_a16w16.py`  (+5/-0)
- `aiter/ops/triton/batched_gemm_bf16.py`  (+4/-0)
- `aiter/ops/triton/extend_attention.py`  (+3/-0)

## Key added lines (kernel files)

**`aiter/ops/triton/batched_gemm_a8w8.py`**
```
from aiter.ops.triton.utils.logger import AiterTritonLogger
_LOGGER = AiterTritonLogger()
_LOGGER.info(
f"BATCHED_GEMM_A8W8: x={tuple(XQ.shape)} w={tuple(WQ.shape)} x_scale={tuple(x_scale.shape)} w_scale={tuple(w_scale.shape
```

**`aiter/ops/triton/batched_gemm_afp4wfp4.py`**
```
from aiter.ops.triton.utils.logger import AiterTritonLogger
_LOGGER = AiterTritonLogger()
_LOGGER.info(
f"BATCHED_GEMM_AFP4WFP4: x={tuple(x.shape)} w={tuple(w.shape)} x_scale={tuple(x.shape)} w_scale={tuple(w.shape)}"
```

**`aiter/ops/triton/batched_gemm_afp4wfp4_pre_quant.py`**
```
from aiter.ops.triton.utils.logger import AiterTritonLogger
_LOGGER = AiterTritonLogger()
_LOGGER.info(
f"BATCHED_GEMM_AFP4WFP_PREQUANT: x={tuple(x.shape)} w={tuple(w.shape)} w_scale={tuple(w.shape)}"
```

**`aiter/ops/triton/batched_gemm_bf16.py`**
```
from aiter.ops.triton.utils.logger import AiterTritonLogger
_LOGGER = AiterTritonLogger()
_LOGGER.info(f"BATCHED_GEMM_BF16: x={tuple(XQ.shape)} w={tuple(WQ.shape)}")
```

**`aiter/ops/triton/extend_attention.py`**
```
from aiter.ops.triton.utils.logger import AiterTritonLogger
_LOGGER = AiterTritonLogger()
```
