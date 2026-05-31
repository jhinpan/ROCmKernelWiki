# Diff summary

- **files changed:** 30
- **lines:** +235 / -4
- **kernel-ish files:** 30

## Files (by churn)

- `aiter/ops/triton/rmsnorm.py`  (+21/-0)
- `aiter/ops/triton/norm.py`  (+20/-0)
- `aiter/ops/triton/mha.py`  (+16/-1)
- `aiter/ops/triton/moe_op_mxfp4.py`  (+11/-0)
- `aiter/ops/triton/fused_qk_concat.py`  (+10/-0)
- `aiter/ops/triton/extend_attention.py`  (+9/-0)
- `aiter/ops/triton/hstu_attention.py`  (+9/-0)
- `aiter/ops/triton/moe_op.py`  (+9/-0)
- `aiter/ops/triton/moe_op_silu_fused.py`  (+9/-0)
- `aiter/ops/triton/moe_op_e2e.py`  (+8/-0)
- `aiter/ops/triton/moe_op_gelu.py`  (+8/-0)
- `aiter/ops/triton/quant.py`  (+7/-1)
- `aiter/ops/triton/fused_mul_add.py`  (+7/-0)
- `aiter/ops/triton/mha_fused_bwd.py`  (+7/-0)
- `aiter/ops/triton/mha_onekernel_bwd.py`  (+7/-0)

## Key added lines (kernel files)

**`aiter/ops/triton/activation.py`**
```
from aiter.ops.triton.utils.logger import AiterTritonLogger
_LOGGER = AiterTritonLogger()
_LOGGER.info(f"ACT_MUL_MXFP4_QUANT: x={tuple(x.shape)} activation={activation}")
```

**`aiter/ops/triton/extend_attention.py`**
```
_LOGGER.info(
f"EXTEND_ATTENTION_FWD: q_extend={tuple(q_extend.shape)} k_extend={tuple(k_extend.shape)} v_extend={tuple(v_extend.shape
+ f"k_buffer={tuple(k_buffer.shape)} v_buffer={tuple(v_buffer.shape)}"
_LOGGER.info(
```

**`aiter/ops/triton/fused_mul_add.py`**
```
from aiter.ops.triton.utils.logger import AiterTritonLogger
_LOGGER = AiterTritonLogger()
_LOGGER.info(
f"FUSED_MUL_ADD: x={tuple(x.shape)} a={tuple(a.shape) if isinstance(a, torch.Tensor) else a} "
```

**`aiter/ops/triton/fused_mxfp4_quant.py`**
```
from aiter.ops.triton.utils.logger import AiterTritonLogger
_LOGGER = AiterTritonLogger()
_LOGGER.info(f"FUSED_RMS_MXFP4_QUANT: inp1={tuple(inp1.shape)}")
_LOGGER.info(f"FUSED_FLATTEN_MXFP4_QUANT: x={tuple(x.shape)}")
```

**`aiter/ops/triton/fused_qk_concat.py`**
```
from aiter.ops.triton.utils.logger import AiterTritonLogger
_LOGGER = AiterTritonLogger()
_LOGGER.info(
f"FUSED_QK_CAT: q1={tuple(q1.shape)} q2={tuple(q2.shape)} k1={tuple(k1.shape)} k2={tuple(k2.shape)} "
```
