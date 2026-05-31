# Diff summary

- **files changed:** 44
- **lines:** +11 / -237
- **kernel-ish files:** 44

## Files (by churn)

- `vllm/model_executor/layers/fused_moe/fused_moe.py`  (+7/-124)
- `vllm/model_executor/layers/fused_moe/modular_kernel.py`  (+1/-19)
- `vllm/model_executor/layers/fused_moe/experts/marlin_moe.py`  (+1/-13)
- `vllm/model_executor/layers/fused_moe/utils.py`  (+0/-10)
- `vllm/model_executor/layers/fused_moe/config.py`  (+0/-6)
- `vllm/model_executor/layers/fused_moe/layer.py`  (+0/-5)
- `vllm/model_executor/layers/fused_moe/oracle/fp8.py`  (+0/-4)
- `tests/kernels/moe/test_deepep_deepgemm_moe.py`  (+0/-3)
- `tests/kernels/moe/test_deepgemm.py`  (+0/-3)
- `tests/kernels/moe/test_flashinfer.py`  (+0/-3)
- `tests/kernels/moe/utils.py`  (+0/-3)
- `vllm/model_executor/layers/fused_moe/oracle/mxfp4.py`  (+0/-3)
- `vllm/model_executor/layers/quantization/quark/quark_moe.py`  (+0/-3)
- `tests/kernels/moe/test_batched_deepgemm.py`  (+0/-2)
- `tests/kernels/moe/test_cutlass_moe.py`  (+0/-2)

## Key added lines (kernel files)

**`vllm/model_executor/layers/fused_moe/experts/marlin_moe.py`**
```
output = torch.empty_like(hidden_states)
```

**`vllm/model_executor/layers/fused_moe/fused_moe.py`**
```
def fused_experts_op(
def fused_experts_op_fake(
op_name="fused_experts",
op_func=fused_experts_op,
```

**`vllm/model_executor/layers/fused_moe/modular_kernel.py`**
```
output = torch.empty_like(hidden_states)
```

**`vllm/model_executor/models/bert_with_rope.py`**
```
final_hidden_states = torch.ops.vllm.fused_experts(
```

**`vllm/model_executor/models/minicpm.py`**
```
hidden_states, self.ws, self.w2s, topk_weights, topk_ids
```
