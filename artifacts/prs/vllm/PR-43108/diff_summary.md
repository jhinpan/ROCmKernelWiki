# Diff summary

- **files changed:** 27
- **lines:** +26 / -148
- **kernel-ish files:** 27

## Files (by churn)

- `vllm/model_executor/layers/fused_moe/prepare_finalize/naive_dp_ep.py`  (+12/-8)
- `vllm/model_executor/layers/fused_moe/experts/cutlass_moe.py`  (+2/-16)
- `tests/kernels/moe/modular_kernel_tools/common.py`  (+10/-5)
- `tests/kernels/moe/modular_kernel_tools/mk_objects.py`  (+0/-15)
- `vllm/model_executor/layers/fused_moe/modular_kernel.py`  (+0/-13)
- `vllm/model_executor/layers/fused_moe/experts/fallback.py`  (+0/-10)
- `vllm/model_executor/layers/fused_moe/fused_moe_modular_method.py`  (+1/-6)
- `vllm/model_executor/layers/fused_moe/experts/cpu_moe.py`  (+0/-6)
- `vllm/model_executor/layers/fused_moe/experts/deep_gemm_moe.py`  (+0/-6)
- `vllm/model_executor/layers/fused_moe/experts/fused_batched_moe.py`  (+0/-6)
- `vllm/model_executor/layers/fused_moe/experts/gpt_oss_triton_kernels_moe.py`  (+0/-6)
- `vllm/model_executor/layers/fused_moe/experts/marlin_moe.py`  (+0/-6)
- `vllm/model_executor/layers/fused_moe/experts/trtllm_mxfp4_moe.py`  (+0/-6)
- `vllm/model_executor/layers/fused_moe/experts/aiter_mxfp4_w4a8_moe.py`  (+0/-3)
- `vllm/model_executor/layers/fused_moe/experts/batched_deep_gemm_moe.py`  (+0/-3)

## Key added lines (kernel files)

**`tests/kernels/moe/modular_kernel_tools/common.py`**
```
if not self.fused_experts_type._supports_current_device():
f"{self.fused_experts_type} not supported on the current device.",
except NotImplementedError:
if config.world_size > 1:
```

**`tests/kernels/moe/test_modular_kernel_combinations.py`**
```
return False
```

**`vllm/model_executor/layers/fused_moe/fused_moe_modular_method.py`**
```
expert_map=layer.expert_map,
```

**`vllm/model_executor/layers/fused_moe/prepare_finalize/naive_dp_ep.py`**
```
) -> tuple[torch.Tensor, list[torch.Tensor] | None, torch.Tensor | None]:
a1q, a1q_scale = moe_kernel_quantize_input(
return a1q, scales, a1q_scale
a1q, scales, a1q_scale_orig = _quantize_and_setup_dispatch(
```
