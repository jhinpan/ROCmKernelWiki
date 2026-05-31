# Diff summary

- **files changed:** 42
- **lines:** +47 / -5
- **kernel-ish files:** 42

## Files (by churn)

- `csrc/include/attention_generic.cuh`  (+2/-1)
- `csrc/include/custom_all_reduce.h`  (+2/-1)
- `csrc/include/pos_encoding.h`  (+2/-1)
- `csrc/include/quant_common.cuh`  (+2/-1)
- `csrc/kernels/attention.cu`  (+2/-1)
- `aiter/dist/communication_op.py`  (+1/-0)
- `aiter/dist/cuda_wrapper.py`  (+1/-0)
- `aiter/dist/custom_all_reduce.py`  (+1/-0)
- `aiter/dist/custom_all_reduce_utils.py`  (+1/-0)
- `aiter/dist/parallel_state.py`  (+1/-0)
- `aiter/dist/shm_broadcast.py`  (+1/-0)
- `aiter/dist/utils.py`  (+1/-0)
- `aiter/fused_moe.py`  (+1/-0)
- `aiter/fused_moe_gelu.py`  (+1/-0)
- `aiter/fused_moe_int8_a8w8.py`  (+1/-0)

## Key added lines (kernel files)

**`csrc/include/attention_generic.cuh`**
```
}  // namespace vllm
```

**`csrc/include/pos_encoding.h`**
```
torch::Tensor &cos_sin_cache_offsets);
```

**`csrc/include/quant_common.cuh`**
```
}  // namespace vllm
```
