# Diff summary

- **files changed:** 28
- **lines:** +1379 / -472
- **kernel-ish files:** 21

## Files (by churn)

- `aiter/dist/utils.py`  (+247/-189)
- `aiter/ops/triton/batched_gemm_a8w8_a_per_token_group_prequant_w_per_batched_tensor_quant.py`  (+337/-0)
- `csrc/ck_gemm_a8w8_blockscale/gemm_a8w8_blockscale.cu`  (+89/-66)
- `aiter/dist/custom_all_reduce_utils.py`  (+75/-55)
- `aiter/dist/custom_all_reduce.py`  (+62/-64)
- `aiter/configs/a8w8_bpreshuffle_untuned_gemm.csv`  (+109/-1)
- `aiter/fused_moe.py`  (+56/-42)
- `aiter/ops/triton/configs/gemm/MI300X-BATCHED_GEMM-A8W8-A_PER_TOKEN_GROUP_PREQUANT_W_PER_BATCHED_TENSOR_QUANT-N=128-K=512.json`  (+81/-0)
- `aiter/ops/triton/configs/gemm/MI300X-BATCHED_GEMM-A8W8-A_PER_TOKEN_GROUP_PREQUANT_W_PER_BATCHED_TENSOR_QUANT-N=512-K=128.json`  (+81/-0)
- `hsa/gfx942/fmoe_2stages/tune.py`  (+57/-7)
- `aiter/configs/a8w8_bpreshuffle_tuned_gemm.csv`  (+48/-0)
- `csrc/ck_gemm_a8w8_bpreshuffle/gemm_a8w8_bpreshuffle_tune.py`  (+28/-9)
- `csrc/ck_gemm_a8w8_bpreshuffle/gemm_a8w8_bpreshuffle.cu`  (+27/-5)
- `csrc/ck_gemm_moe_2stages_codegen/gemm_moe_ck2stages.h`  (+20/-0)
- `aiter/ops/quant.py`  (+8/-8)

## Key added lines (kernel files)

**`aiter/dist/custom_all_reduce.py`**
```
from .custom_all_reduce_utils import gpu_p2p_access_check
from .utils import get_cuda_visible_devices
return inp.is_contiguous() or (
inp.storage().nbytes() - inp.storage_offset() * inp.element_size()
```

**`aiter/dist/custom_all_reduce_utils.py`**
```
from .utils import (
cuda_device_count_stateless,
update_environment_variables,
get_cuda_visible_devices,
```

**`aiter/dist/parallel_state.py`**
```
if "HIP_VISIBLE_DEVICES" not in os.environ:
update_environment_variables(
{"HIP_VISIBLE_DEVICES": (",".join(map(str, range(world_size))))}
```

**`aiter/dist/utils.py`**
```
from typing import (
AsyncGenerator,
Awaitable,
Callable,
```

**`aiter/fused_moe.py`**
```
quant_type = quant_remap.get(quant_type, quant_type)
quant_remap = {QuantType.per_128x128: QuantType.per_1x128}
run_1stage = False
not doweight_stage1
```
