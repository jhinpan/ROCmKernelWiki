# Diff summary

- **files changed:** 89 (diff was byte-capped; summary is partial)
- **lines:** +3745 / -4563
- **kernel-ish files:** 21

## Files (by churn)

- `aiter/ops/triton/configs/gemm/gfx950-GEMM-A16W16-N=128-K=2880.json`  (+144/-144)
- `aiter/ops/triton/configs/gemm/gfx950-GEMM-A16W16-N=2880-K=4096.json`  (+144/-144)
- `aiter/ops/triton/configs/gemm/gfx950-GEMM-A16W16-N=2880-K=512.json`  (+144/-144)
- `aiter/ops/triton/configs/gemm/gfx950-GEMM-A16W16-N=5120-K=2880.json`  (+144/-144)
- `aiter/ops/triton/configs/gemm/gfx950-GEMM-A16W16-N=640-K=2880.json`  (+144/-144)
- `aiter/ops/triton/configs/gemm/gfx950-FUSED-GEMM-A8W8_BLOCKSCALE-A16W16-N8=512-N16=256-K=7168.json`  (+108/-108)
- `aiter/ops/triton/configs/gemm/gfx950-GEMM-A16W16-N=256-K=7168.json`  (+91/-91)
- `aiter/ops/triton/configs/gemm/gfx950-GEMM-A16W8_BLOCKSCALE-N=7168-K=2048.json`  (+84/-85)
- `aiter/ops/triton/configs/gemm/gfx950-GEMM-A16WFP4.json`  (+84/-85)
- `aiter/ops/triton/configs/gemm/gfx950-FUSED-GEMM-AFP4WFP4-A16W16-N4=512-N16=256-K=7168.json`  (+84/-84)
- `aiter/ops/triton/configs/gemm/gfx950-GEMM-A16WFP4-N=7168-K=2048.json`  (+84/-84)
- `aiter/ops/triton/configs/gemm/gfx950-BATCHED_GEMM-AFP4WFP4.json`  (+78/-80)
- `aiter/ops/triton/configs/gemm/gfx950-BATCHED_GEMM_PREQUANT-AFP4WFP4.json`  (+78/-80)
- `aiter/ops/triton/configs/gemm/gfx950-GEMM-A16W16-ATOMIC-N=256-K=7168.json`  (+79/-79)
- `aiter/ops/triton/configs/gemm/gfx950-GEMM-A8WFP4.json`  (+78/-80)

## Key added lines (kernel files)

**`aiter/ops/triton/_triton_kernels/batched_gemm_a16wfp4.py`**
```
from ..utils.gemm_config_utils import get_gemm_config
return get_gemm_config("BATCHED_GEMM_PREQUANT-AFP4WFP4", M, N, 2 * K)
```

**`aiter/ops/triton/_triton_kernels/batched_gemm_a8w8.py`**
```
from ..utils.gemm_config_utils import get_gemm_config
return get_gemm_config("BATCHED_GEMM-A8W8", M, N, K)
```

**`aiter/ops/triton/_triton_kernels/batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant.py`**
```
from ..utils.gemm_config_utils import get_gemm_config
return get_gemm_config(
"BATCHED_GEMM-A8W8-A_PER_TOKEN_GROUP_PREQUANT_W_PER_BATCHED_TENSOR_QUANT",
```

**`aiter/ops/triton/_triton_kernels/batched_gemm_afp4wfp4.py`**
```
from ..utils.gemm_config_utils import get_gemm_config
config = get_gemm_config("BATCHED_GEMM-AFP4WFP4", M, N, 2 * K)
```

**`aiter/ops/triton/_triton_kernels/batched_gemm_bf16.py`**
```
from ..utils.gemm_config_utils import get_gemm_config
return get_gemm_config("BATCHED_GEMM-A16W16", M, N, K)
```
