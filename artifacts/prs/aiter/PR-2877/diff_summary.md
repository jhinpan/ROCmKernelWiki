# Diff summary

- **files changed:** 14
- **lines:** +852 / -827
- **kernel-ish files:** 8

## Files (by churn)

- `aiter/ops/flydsl/kernels/splitk_hgemm.py`  (+382/-344)
- `aiter/ops/flydsl/gemm_kernels.py`  (+113/-129)
- `aiter/ops/flydsl/kernels/small_m_hgemm.py`  (+107/-123)
- `aiter/configs/model_configs/llama70B_bf16_tuned_gemm.csv`  (+48/-48)
- `aiter/configs/model_configs/kimik2_bf16_tuned_gemm.csv`  (+46/-46)
- `aiter/configs/model_configs/llama405B_bf16_tuned_gemm.csv`  (+43/-43)
- `aiter/configs/model_configs/qwen32B_bf16_tuned_gemm.csv`  (+34/-34)
- `aiter/configs/model_configs/glm5_bf16_tuned_gemm.csv`  (+25/-25)
- `aiter/configs/model_configs/dsv3_bf16_tuned_gemm.csv`  (+22/-22)
- `aiter/aot/flydsl/gemm.py`  (+17/-6)
- `aiter/ops/flydsl/kernels/tensor_shim.py`  (+7/-2)
- `aiter/ops/flydsl/kernels/hgemm_dispatch.py`  (+5/-2)
- `gradlib/gradlib/GemmTuner.py`  (+2/-2)
- `aiter/tuned_gemm.py`  (+1/-1)

## Key added lines (kernel files)

**`aiter/aot/flydsl/gemm.py`**
```
from aiter.ops.flydsl.gemm_kernels import (
SPLIT_K_SEMAPHORE_MAX_LEN,
get_flydsl_splitk_hgemm_kernel_params,
semaphore = torch.zeros(
```

**`aiter/ops/flydsl/gemm_kernels.py`**
```
import flydsl.expr as fx
SPLIT_K_SEMAPHORE_MAX_LEN = 256
SPLIT_K_GLOBAL_SIGNAL: dict[SplitKStreamKey, torch.Tensor] = {}
HGEMM_TILE_N_OPTIONS = (64, 128, 256)
```

**`aiter/ops/flydsl/kernels/hgemm_dispatch.py`**
```
b_preshuffle: bool = False,
if b_preshuffle:
raise ValueError(
"Generic FlyDSL HGEMM does not support `b_preshuffle=True`"
```

**`aiter/ops/flydsl/kernels/small_m_hgemm.py`**
```
semaphore: fx.Tensor,
signal: fx.Tensor,
if const_expr(IS_SPLIT_K):
smem_bc_ptr = SmemPtr(base_ptr, smem_a_offset, T.i32, shape=(1,))
```

**`aiter/ops/flydsl/kernels/splitk_hgemm.py`**
```
from flydsl.expr import (
buffer_ops,
const_expr,
range_constexpr,
```
