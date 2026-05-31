# Diff summary

- **files changed:** 17
- **lines:** +83 / -98
- **kernel-ish files:** 17

## Files (by churn)

- `vllm/model_executor/layers/fused_moe/router/base_router.py`  (+27/-27)
- `vllm/model_executor/layers/fused_moe/layer.py`  (+29/-20)
- `vllm/model_executor/layers/fused_moe/router/router_factory.py`  (+2/-13)
- `tests/kernels/moe/test_routing.py`  (+4/-7)
- `vllm/distributed/eplb/eplb_state.py`  (+11/-0)
- `vllm/model_executor/layers/fused_moe/router/aiter_shared_routed_fused_moe_router.py`  (+1/-3)
- `vllm/model_executor/layers/fused_moe/router/custom_routing_router.py`  (+1/-3)
- `vllm/model_executor/layers/fused_moe/router/fused_topk_bias_router.py`  (+1/-3)
- `vllm/model_executor/layers/fused_moe/router/fused_topk_router.py`  (+1/-3)
- `vllm/model_executor/layers/fused_moe/router/grouped_topk_router.py`  (+1/-3)
- `vllm/model_executor/layers/fused_moe/router/routing_simulator_router.py`  (+1/-3)
- `vllm/model_executor/layers/fused_moe/router/zero_expert_router.py`  (+1/-3)
- `vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe/compressed_tensors_moe_w4a8_fp8.py`  (+0/-4)
- `tests/distributed/test_eplb_fused_moe_layer_dep_nvfp4.py`  (+2/-1)
- `tests/kernels/moe/test_moe_layer.py`  (+0/-3)

## Key added lines (kernel files)

**`tests/distributed/test_eplb_fused_moe_layer_dep_nvfp4.py`**
```
from vllm.distributed.eplb.eplb_state import EplbLayerState
fml.eplb_state = EplbLayerState()
```

**`tests/kernels/moe/test_routing.py`**
```
def setup_eplb_state(
enable_eplb: bool, global_num_experts: int
) -> EplbLayerState | None:
return None
```

**`vllm/distributed/eplb/eplb_state.py`**
```
def set_layer_state(
moe_layer_idx: int,
expert_load_view: torch.Tensor,
logical_to_physical_map: torch.Tensor,
```

**`vllm/model_executor/layers/fused_moe/layer.py`**
```
self.eplb_state: EplbLayerState | None = None
if enable_eplb:
if self.use_ep and self.global_num_experts % self.ep_size != 0:
raise ValueError(
```

**`vllm/model_executor/layers/fused_moe/router/aiter_shared_routed_fused_moe_router.py`**
```
eplb_state: EplbLayerState | None = None,
```
