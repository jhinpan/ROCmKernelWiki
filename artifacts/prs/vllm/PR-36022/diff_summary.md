# Diff summary

- **files changed:** 19
- **lines:** +418 / -44
- **kernel-ish files:** 17

## Files (by churn)

- `vllm/model_executor/layers/fused_moe/flashinfer_nvlink_one_sided_prepare_finalize.py`  (+146/-0)
- `vllm/distributed/device_communicators/all2all.py`  (+132/-6)
- `tests/kernels/moe/modular_kernel_tools/mk_objects.py`  (+38/-5)
- `vllm/model_executor/layers/fused_moe/all2all_utils.py`  (+21/-4)
- `vllm/distributed/device_communicators/cuda_communicator.py`  (+17/-4)
- `vllm/model_executor/layers/fused_moe/config.py`  (+16/-4)
- `vllm/distributed/device_communicators/mnnvl_compat.py`  (+7/-7)
- `vllm/utils/flashinfer.py`  (+11/-2)
- `vllm/config/parallel.py`  (+5/-2)
- `vllm/model_executor/layers/fused_moe/deep_gemm_moe.py`  (+4/-1)
- `vllm/model_executor/layers/fused_moe/fused_marlin_moe.py`  (+4/-1)
- `vllm/model_executor/layers/fused_moe/fused_moe.py`  (+4/-1)
- `vllm/model_executor/layers/fused_moe/rocm_aiter_fused_moe.py`  (+4/-1)
- `docs/design/moe_kernel_features.md`  (+2/-1)
- `docs/serving/expert_parallel_deployment.md`  (+2/-1)

## Key added lines (kernel files)

**`tests/kernels/moe/modular_kernel_tools/mk_objects.py`**
```
from vllm.utils.flashinfer import (
has_flashinfer_cutlass_fused_moe,
has_flashinfer_nvlink_one_sided,
from vllm.model_executor.layers.fused_moe.flashinfer_nvlink_two_sided_prepare_finalize import (  # noqa: E501
```

**`vllm/config/parallel.py`**
```
"flashinfer_all2allv",  # temporary alias for flashinfer_nvlink_two_sided
"flashinfer_nvlink_two_sided",
"flashinfer_nvlink_one_sided",
- "flashinfer_nvlink_two_sided": Use flashinfer two-sided kernels for mnnvl
```

**`vllm/distributed/device_communicators/all2all.py`**
```
import torch.distributed as dist
from vllm.utils.flashinfer import (
has_flashinfer_nvlink_one_sided,
has_flashinfer_nvlink_two_sided,
```

**`vllm/distributed/device_communicators/cuda_communicator.py`**
```
self.all2all_backend == "flashinfer_all2allv"
or self.all2all_backend == "flashinfer_nvlink_two_sided"
if self.all2all_backend == "flashinfer_all2allv":
logger.warning_once(
```

**`vllm/distributed/device_communicators/mnnvl_compat.py`**
```
from vllm.utils.flashinfer import has_flashinfer_nvlink_two_sided
assert has_flashinfer_nvlink_two_sided(), "Flashinfer alltoallv module cannot be found"
obj_list = [data]
dist.broadcast_object_list(obj_list, src=root, group=self._group)
```
